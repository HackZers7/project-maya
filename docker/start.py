#!/usr/bin/env python3
"""Starts Project Maya in the container. The engine, the vision encoder, Maya's Python
environment and llama.cpp's source are baked into the image (docker/build.py), so a start
downloads and compiles nothing: the first run of a profile only writes the pack beside the
model files (if it is not there yet) and the profile's configuration in /state; every later
run starts straight from that configuration.

The server stays up without the model (MAYA_START_UNLOADED=1, set in docker-compose.yml):
no engine process, no vision encoder, the GPUs and the RAM tier free. The model loads on
POST /models/load or on the first request (Service.ensure_engine in serve/server.py) and
unloads on POST /models/unload, after MAYA_IDLE_UNLOAD_S seconds idle, or on a request's
keep_alive: 0. The model management endpoints are llama.cpp's shape (on by default since
Maya v1.1.47): /models/load, /models/unload, /models/sse - a client like Open WebUI or a
load balancer drives them. Loading a different installed model than the running one
switches the server to it (a 503 + Retry-After answers while it restarts into that model).

MAYA_MODEL is the model the server starts with and the bare /models/load (no name) loads:
"glm-5.3-flash" (GLM's Maya-M quant) or "hy4-flash" (Tencent's Hy4-Flash, AngelSlim's
Q4_K_M). Only one model runs at a time (both cards are needed by either); switching no
longer needs a container recreate - /models/load with the other model's name does it.
Both models' configurations are kept in /state and refreshed from MAYA_* at every start,
so a runtime switch always lands on a current configuration.

The server and its engine run in a session of their own, so a docker stop reaches the whole
tree as one group signal - the same as Ctrl+C in a terminal, which is how Maya is meant to
be stopped (the engine then saves its state and exits cleanly). This script runs as PID 1:
it stays up until the tree is gone, so the engine finishes saving its state instead of
being SIGKILLed mid-save.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.environ.get("MAYA_ROOT", "/opt/project-maya")
STATE = "/state"

# the knobs, with the image's own defaults: docker-compose.yml sets most of them (the
# comments there say what each does); anything unset takes the value below
DEFAULTS = {
    "MAYA_PORT": "8080",
    "MAYA_GPUS": "0,1",                    # the GPUs to split the model's layers across, as nvidia-smi numbers them
    "MAYA_CONTEXT": "32768",               # context length in tokens (the model handles up to 1M)
    "MAYA_DRAFT": "mtp",                   # "mtp": the model's own draft block; "dflash2": a separate drafter,
                                           # downloaded into the models folder at the first setup (0.7 GB)
    "MAYA_RAM_GB": "56",                   # the engine's pinned RAM tier, in GB
    "MAYA_VISION": "yes",                  # "no": the model reads text only
    "MAYA_VISION_THREADS": "8",            # CPU threads of the image encoder (it runs in RAM, not on a GPU)
    "MAYA_KV_STREAMING": "on",             # "on": the attention cache lives in RAM with a VRAM window (v1.1.6+)
    "MAYA_KV_RESIDENT": "65536",           # KV streaming's window, in context positions, kept in VRAM
    "MAYA_PREFILL": "auto",                # the shared prefill chunk, in tokens, that a model takes while its own
                                           # knob below is unset (kept so an older compose file keeps working)
    "MAYA_GLM_PREFILL": "",                # GLM's prefill chunk (--prefill): unset = MAYA_PREFILL; "auto" is the
                                           # engine's own sizing (8192 on this split)
    "MAYA_HY4_PREFILL": "",                # Hy4's prefill chunk (maya-hy4's --chunk): unset = MAYA_PREFILL; "auto"
                                           # is the engine plan's (4096 a step here); a number stages the chunk's
                                           # rows in RAM in proportion to it
    "MAYA_PARALLEL": "1",                  # how many requests decode at once (1-8): a maya.py switch
    "MAYA_ABLATE": "off",                  # "on": the abliterated model (the vector is <GGUF tree>/vector/<file>);
                                           # "off": the stock model; anything else: a vector file's path to use
    "MAYA_IDLE_UNLOAD_S": "900",           # the model unloads itself after this many seconds without a request
    "MAYA_MANAGE_ALLOW": "127.0.0.1",        # extra client IPs that may pull, delete and switch models without
                                            # an API key, comma-separated. Loopback is already "this PC" for
                                            # the gate, so the default bakes nothing: a rootless Docker sets
                                            # 10.0.2.2 here (slirp4netns' address for the host - this PC's own
                                            # requests arrive with that source); rootful Docker and bare
                                            # metal never need it
    "MAYA_GLM_SLOT_KEEP": "1",             # "1": the conversation slots outlive the engine
    "MAYA_GLM_SLOTS": "16",                # conversations kept aside on the disk (the engine's default is 4)
    "MAYA_GLM_SLOT_GB": "64",              # their total size on disk, GB (the engine's 16)
    "MAYA_GLM_SLOT_DIR": "/data/slots",    # the slots folder (mounted from ./data/slots; empty: the engine's
                                           # default <pack>/slots, beside the GGUF shards in the models mount)
    "MAYA_MODEL": "glm-5.3-flash",         # the default model: "glm-5.3-flash" or "hy4-flash"
}
env = {k: os.environ.setdefault(k, v) for k, v in DEFAULTS.items()}

say = lambda *a: print(*a, flush=True)   # noqa: E731  every line here goes to `docker logs`


def set_flag(args: list[str], flag: str, value=None) -> list[str]:
    """Set or remove a flag (always written with its value) in the engine's args."""
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2   # the flag and its value
        else:
            out.append(args[i])
            i += 1
    if value is not None:
        out += [flag, str(value)]
    return out


def kv_flags(args: list[str]) -> list[str]:
    """The KV streaming pair in the engine's args (the setup knows no --kv-* switches, so
    they are written here at every start): on keeps the attention cache in RAM with
    MAYA_KV_RESIDENT positions of it in VRAM (the freed VRAM holds routed experts); off
    keeps all of the KV in VRAM."""
    on = env["MAYA_KV_STREAMING"] == "on"
    resident = env["MAYA_KV_RESIDENT"]
    args = set_flag(args, "--kv-streaming", "on" if on else "off")
    if on and resident:
        return set_flag(args, "--kv-resident", int(resident))
    return set_flag(args, "--kv-resident")


def prefill(model: str) -> str:
    """The model's (GLM / HY4) prefill chunk: its own knob wins, else the shared MAYA_PREFILL
    (an older compose file sets that alone), else auto - the engine's own sizing."""
    return env[f"MAYA_{model}_PREFILL"] or env["MAYA_PREFILL"] or "auto"


def seed_data() -> None:
    """The engines keep what they measure on this PC (the CPU lanes' timings, the prompt
    routes, the 4-bit copies' verdict - roughly three minutes at a first start) in data/,
    mounted from ./data/maya so that a container recreate does not measure again. The
    image's own part of it (.maya-data-image: the expert profiles and the seeds, which are
    never measured back) is copied in at every start; the measurements, absent from the
    image, are left as they are."""
    image_data = os.path.join(ROOT, ".maya-data-image")
    if os.path.isdir(image_data):
        os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
        shutil.copytree(image_data, os.path.join(ROOT, "data"), dirs_exist_ok=True, symlinks=True)


def run_setup(args: list[str]) -> None:
    """The first-time setup of a model, unattended: ./maya.sh --setup writes the
    configuration this start then keeps and refreshes. The pack is built beside the model
    files if missing; nothing is downloaded into the container (--models-dir points at the
    mounted GGUF tree)."""
    subprocess.run(["./maya.sh", "--setup", "--plain", "--yes", "--no-start", *args], check=True)


def manage_hy4(cfg: str) -> None:
    """Hy4-Flash (Tencent's, AngelSlim's Q4_K_M): configuration managed at every start.
    When the model files are there, the configuration exists and is refreshed from MAYA_*;
    without the files the section is skipped unless MAYA_MODEL asks for Hy4-Flash - the
    server can still run GLM, and a /models/load for hy4-flash would fail with the reason
    until the files are mounted."""
    hy4_root = "/models/AngelSlim/Hy4-flash-GGUF"
    hy4_dir = os.path.join(hy4_root, "Hy4-flash")
    if not os.path.isdir(hy4_dir):
        if env["MAYA_MODEL"] == "hy4-flash":
            sys.exit(f"[entrypoint] MAYA_MODEL=hy4-flash, but {hy4_dir} does not exist - the Q4_K_M set must be "
                     "there: the two GGUF shards (the model, and the n-gram table) and mmproj-Hy4-flash-bf16.gguf")
        say(f"[entrypoint] note: {hy4_dir} does not exist - the Hy4-Flash configuration is not managed;"
            "\n[entrypoint]       a /models/load naming hy4-flash will fail until the Q4_K_M set is mounted")
        return

    if not os.path.isfile(cfg):
        say(f"[entrypoint] no Hy4-Flash configuration in {STATE}: writing it (the maya-hy4 engine is in the\n"
            "[entrypoint] image; the pack is built beside the model files if missing)")
        if int(env["MAYA_CONTEXT"]) > 262144:
            say(f"[entrypoint] note: MAYA_CONTEXT={env['MAYA_CONTEXT']} is past Hy4-Flash's tested 262144 "
                "(upstream ran\n[entrypoint] this set end to end up to 256K) - using it as set")
        # MAYA_DRAFT is shared with GLM: "mtp" is the Q4_K_M set's own MTP block (it drafts,
        # the model checks - the same answers), "off" no drafts; the Q4_K_M has no DFly2
        # drafter, so GLM's "dflash2" does not translate
        drafting = {"mtp": "mtp", "off": "off"}.get(env["MAYA_DRAFT"])
        if drafting is None:
            sys.exit(f"[entrypoint] MAYA_DRAFT={env['MAYA_DRAFT']} does not apply to Hy4-Flash's Q4_K_M "
                     "(no DFly2 drafter in the set): use mtp or off")
        setup_args = ["--model", "hy4-flash-q4_k_m",
                      "--model-dir", hy4_dir,
                      "--models-dir", hy4_root,
                      "--gpus", env["MAYA_GPUS"],
                      "--context", env["MAYA_CONTEXT"],
                      "--drafting", drafting,
                      "--host", "0.0.0.0",
                      "--port", env["MAYA_PORT"]]
        if env["MAYA_VISION"] != "yes":
            setup_args.append("--no-vision")
        run_setup(setup_args)
        shutil.copy(os.path.join(ROOT, "maya-hy4-flash.json"), cfg)
        say(f"[entrypoint] configuration kept in {cfg}")

    # the engine flags applied to the kept config at every start: the context from
    # MAYA_CONTEXT, KV streaming, the prompt chunk - maya-hy4's own --chunk (MAYA_HY4_PREFILL,
    # the shared MAYA_PREFILL while it is unset; auto: the engine's plan picks it, 4096 a
    # step here; a number sets it, and the chunk's staged rows sit in RAM in proportion to
    # it). idle_unload_s is the inactivity timeout after
    # which the model unloads itself (MAYA_IDLE_UNLOAD_S). The GLM-only pins (the RAM tier,
    # the slots, the ablation vector) do not apply to maya-hy4: its experts come from the
    # model file's page cache, which the container's memory limit bounds by itself.
    with open(cfg) as f:
        c = json.load(f)
    c["idle_unload_s"] = int(env["MAYA_IDLE_UNLOAD_S"] or 0)
    args = c.get("args") or []
    args = set_flag(args, "--max-context", int(env["MAYA_CONTEXT"]))
    args = kv_flags(args)
    chunk = prefill("HY4")
    args = set_flag(args, "--chunk", int(chunk) if chunk != "auto" else None)
    c["args"] = args
    with open(cfg, "w") as f:
        json.dump(c, f, indent=1)


def manage_glm(cfg: str) -> None:
    """GLM-5.3-Flash (the default): configuration managed at every start. The kept config's
    file name is the name the model management API shows and accepts (serve/model_api.py:
    maya-<name>.json -> "<name>"), so it matches MAYA_MODEL: a load or a switch can name the
    model "glm-5.3-flash"."""
    glm_root = "/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF"
    quant = "Maya-M"    # the quant folder of the GGUF tree that runs (the one on this machine)
    quant_dir = os.path.join(glm_root, quant)
    if not os.path.isdir(quant_dir):
        if env["MAYA_MODEL"] == "glm-5.3-flash":
            listing = ""
            try:
                listing = ", ".join(sorted(os.listdir(glm_root)))
            except OSError:
                pass
            sys.exit(f"[entrypoint] MAYA_MODEL=glm-5.3-flash, but {quant_dir} does not exist;"
                     f" the folders of the GGUF tree: {listing or 'unreadable'}")
        if env["MAYA_MODEL"] != "hy4-flash":
            sys.exit(f"[entrypoint] MAYA_MODEL={env['MAYA_MODEL']} is neither glm-5.3-flash nor hy4-flash, "
                     f"and {quant_dir} does not exist")
        say(f"[entrypoint] note: {quant_dir} does not exist - the GLM configuration is not managed;"
            "\n[entrypoint]       a /models/load naming glm-5.3-flash will fail until it is mounted")
        return

    # the config kept under the old name (by the quant) becomes the one under MAYA_MODEL's name
    old_cfg = os.path.join(STATE, f"maya-{quant}.json")
    if not os.path.isfile(cfg) and os.path.isfile(old_cfg):
        os.replace(old_cfg, cfg)
        say(f"[entrypoint] renamed the kept GLM configuration to {cfg} (the name /models/load shows)")

    # the vision files (the vision tower, its tokenizer) are shared by every quant, but the
    # setup looks for them in the model's own folder (<model folder>/vision): link the
    # shared folder in
    if env["MAYA_VISION"] == "yes":
        shared = os.path.join(glm_root, "vision")
        link = os.path.join(quant_dir, "vision")
        if os.path.isdir(shared) and not os.path.exists(link):
            try:
                os.symlink(os.path.relpath(shared, quant_dir), link)
            except OSError as e:
                sys.exit(f"[entrypoint] could not link the shared vision folder into {quant_dir}: {e}")

    if not os.path.isfile(cfg):
        say(f"[entrypoint] no configuration for {quant} in {STATE}: writing it (the engine, the\n"
            "[entrypoint] vision encoder and llama.cpp's source are in the image; the pack is built\n"
            "[entrypoint] beside the model files if missing)")
        # --models-dir "$GLM_ROOT": what the setup downloads (the DFlash2 drafter) lands in
        # the mounted GGUF tree, not the container. --no-speed-calibration: an unattended
        # start keeps the drafter it was told to use instead of measuring 5-20 min and
        # possibly switching back to the MTP block - measure by hand with:
        # ./maya.sh --speed-calibration
        setup_args = ["--gguf-dir", quant_dir,
                      "--models-dir", glm_root,
                      "--gpus", env["MAYA_GPUS"],
                      "--context", env["MAYA_CONTEXT"],
                      "--draft", env["MAYA_DRAFT"],
                      "--no-speed-calibration",
                      "--host", "0.0.0.0",
                      "--port", env["MAYA_PORT"]]
        if env["MAYA_VISION"] != "yes":
            setup_args.append("--no-vision")
        run_setup(setup_args)
        shutil.copy(os.path.join(ROOT, f"maya-{quant.lower()}.json"), cfg)
        say(f"[entrypoint] configuration kept in {cfg}")

    # /proc/meminfo inside a container reports the host's RAM, not the container's limit, so
    # the engine would size its tier from ~80 free GB and get OOM-killed: pin the tier to
    # MAYA_RAM_GB here (also after a later change of the setting, which the setup would not
    # see). The image encoder is pinned to the CPU: gpu:false starts strata-vision without
    # --gpu, so it holds its ~1.2 GB in RAM and never touches the model's GPUs or their
    # expert cache. MAYA_ABLATE pins the ablation vector in the env section the same way:
    # the engine reads STRATA_GLM_ABLATE (the file) and STRATA_GLM_ABLATE_LAYERS (the
    # layers the file holds, parsed from its name, as tools/glm_ablate_config.py does) at
    # every start - so it can be turned on or off without re-running a setup. A missing
    # file ends the start here: the engine would only log a warning and run the stock model
    # instead. The slots (MAYA_GLM_SLOT_KEEP/MAYA_GLM_SLOTS/MAYA_GLM_SLOT_GB/MAYA_GLM_SLOT_DIR)
    # are pinned the same way (docs/SETTINGS.md: MAYA_GLM_SLOT_KEEP and the slots row); an
    # empty MAYA_GLM_SLOT_DIR restores the engine's <pack>/slots. idle_unload_s is the
    # inactivity timeout after which the model unloads itself (MAYA_IDLE_UNLOAD_S).
    with open(cfg) as f:
        c = json.load(f)
    c["idle_unload_s"] = int(env["MAYA_IDLE_UNLOAD_S"] or 0)
    c.setdefault("env", {})
    c["env"]["STRATA_GLM_RAM_GB"] = env["MAYA_RAM_GB"]
    c["env"]["MAYA_GLM_SLOT_KEEP"] = env["MAYA_GLM_SLOT_KEEP"]
    c["env"]["MAYA_GLM_SLOTS"] = env["MAYA_GLM_SLOTS"]
    c["env"]["MAYA_GLM_SLOT_GB"] = env["MAYA_GLM_SLOT_GB"]
    if env["MAYA_GLM_SLOT_DIR"]:
        c["env"]["MAYA_GLM_SLOT_DIR"] = env["MAYA_GLM_SLOT_DIR"]
    else:
        c["env"].pop("MAYA_GLM_SLOT_DIR", None)
    for k in ("STRATA_GLM_ABLATE", "STRATA_GLM_ABLATE_LAYERS", "STRATA_GLM_ABLATE_SCALE"):
        c["env"].pop(k, None)
    a = env["MAYA_ABLATE"].strip()
    if a and a.lower() not in ("off", "no", "0", "false"):
        vec = a
        if a.lower() in ("on", "yes", "1", "true"):
            vec = os.path.join(glm_root, "vector", "glm53-refusal-per-block-3dir.layers45.f32")
        if not os.path.isfile(vec):
            sys.exit(f"entrypoint: MAYA_ABLATE={a}: {vec} is not there - the engine would run the stock model "
                     "silently; put the vector in the models mount or set MAYA_ABLATE=off")
        c["env"]["STRATA_GLM_ABLATE"] = vec
        m = re.search(r"\.layers(\d+)\.", os.path.basename(vec))
        if m:
            c["env"]["STRATA_GLM_ABLATE_LAYERS"] = m.group(1)
        say(f"[entrypoint] ablation on: {vec}" + (f", layers {c['env']['STRATA_GLM_ABLATE_LAYERS']}" if m else ""))
    if env["MAYA_VISION"] == "yes" and c.get("vision"):
        c["vision"]["gpu"] = False
        c["vision"]["threads"] = int(env["MAYA_VISION_THREADS"])
    args = c.get("args") or []
    args = kv_flags(args)
    # the prefill chunk: the engine's --prefill flag, pinned like the KV pair so it can be
    # changed any time (prefill(): the model's own knob, else MAYA_PREFILL, else "auto" -
    # the engine's own sizing, 8192 on this split)
    args = set_flag(args, "--prefill", prefill("GLM"))
    c["args"] = args
    with open(cfg, "w") as f:
        json.dump(c, f, indent=1)


def start_server(cfg: str, start_args: list[str]) -> int:
    """The server and its engine in a session of their own: a docker stop reaches the whole
    tree as one group signal, and this PID 1 stays up until the tree is gone - the engine
    finishes saving its state instead of being SIGKILLed mid-save. start_args: the default
    model's own extra start flags (GLM's --parallel; maya-hy4 serves one request at a time
    and takes none)."""
    say(f"[entrypoint] starting Maya on port {env['MAYA_PORT']} (with MAYA_START_UNLOADED=1 the model loads "
        "on the first request or POST /models/load)")
    child = subprocess.Popen(["./maya.sh", "--config", cfg, "--host", "0.0.0.0", "--port", env["MAYA_PORT"],
                              *start_args], start_new_session=True)
    pgid = child.pid      # start_new_session: the child leads its own process group

    def forward(signum, _frame):    # the whole tree gets the signal, as one Ctrl+C would
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    rc = child.wait()
    # `wait` returns as soon as a signal interrupts the child, and the child itself dies at
    # once - but the tree it started (the server, the engine: the child's whole session)
    # runs its own graceful shutdown, and PID 1 must stay up until that tree is gone. The
    # group is probed with killpg(signal 0) - the real syscall, unlike dash's builtin; the
    # re-parented children are reaped along the way (a zombie would keep the loop alive
    # forever).
    while True:
        try:
            os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    return 128 - rc if rc < 0 else rc


def main() -> int:
    os.chdir(ROOT)
    seed_data()
    say(f"[entrypoint] Project Maya {read_image_ref()}, default model: {env['MAYA_MODEL']}")

    glm_cfg = os.path.join(STATE, "maya-glm-5.3-flash.json")
    hy4_cfg = os.path.join(STATE, "maya-hy4-flash.json")
    manage_hy4(hy4_cfg)
    manage_glm(glm_cfg)

    # the config the server starts with: MAYA_MODEL is the default model. START_ARGS: the
    # default model's own extra start flags
    if env["MAYA_MODEL"] == "hy4-flash":
        cfg, start_args = hy4_cfg, []          # maya-hy4 serves one request at a time
    else:
        cfg, start_args = glm_cfg, ["--parallel", env["MAYA_PARALLEL"]]
    return start_server(cfg, start_args)


def read_image_ref() -> str:
    """which Maya this image carries (written by docker/build.sh at image build time)"""
    try:
        with open(os.path.join(ROOT, ".maya-image-ref")) as f:
            return f.read().strip()
    except OSError:
        return "unknown version"


if __name__ == "__main__":
    sys.exit(main())
