#!/usr/bin/env python3
"""The image build's one setup step: everything a container start would otherwise compile or
download is done here, at image build time, through Maya's own functions - so a start finds
every stamp in place and does nothing but write the pack and the configuration.

    python3 docker/build.py [--stage {all,env,engine,vision,hy4}]

    all (default)   env -> engine -> vision -> hy4, in order
    env             Maya's Python environment (.venv: pip's PY_PACKAGES) and llama.cpp's
                    source, fetched at the commit setup.py pins
    engine          the GLM engine (`maya`, target maya: GLM-5.3-Flash, GLM-5.3 and
                    Qwen-3.8-Flash-Next) for CUDA_ARCHS - and its stamp build/MAYA-BUILD.json,
                    which the runtime setup reads to report everything as already done
    vision          the vision encoder (llama.cpp's mtmd, build-vision/bin/maya-vision) from
                    the engine's stamp: always the same cards the engine was built for
    hy4             the Hy4-Flash engine (`maya-hy4`) beside the GLM one: the same build/
                    folder and stamp keep both targets, so only its own sources compile

The script is self-contained about the environment: run with a plain python3 it creates
ROOT/.venv first (pip's packages and the imports below need it), installs Maya's packages
into it and re-runs itself with the venv's interpreter - the way docker/Dockerfile calls it.
CUDA_ARCHS (the GPU architectures to compile for, ";"-separated; the older ones below 75
need the GPUs at build time, which a build container has not) and MAYA_GPUS (only recorded
in the stamp) come from the environment. No GPU is needed here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MAYA_ROOT") or Path(__file__).resolve().parent.parent)

# Maya's own installer and the build's stamps live in the repository this script ships in
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def venv_python() -> Path:
    return ROOT / ".venv" / "bin" / "python"


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()


def stage_env() -> int:
    """Maya's Python environment and llama.cpp's source (the two things pip and the engine
    build need). Runs inside the venv - ensure_venv() has re-exec'd into it by now."""
    import setup as S     # noqa: PLC0415  the repository's own installer
    import maya           # noqa: PLC0415  PY_PACKAGES

    llama = S.get_llama_cpp()
    S.pip_install(maya.PY_PACKAGES, ", ".join(maya.PY_PACKAGES))
    print(f"build[{ROOT}]: .venv ready, llama.cpp at {llama}")
    return 0


def archs_and_gpus() -> tuple[list[int], list[int]]:
    archs = [int(a) for a in os.environ.get("CUDA_ARCHS", "86;120").replace(",", ";").split(";") if a.strip()]
    gpus = [int(g) for g in os.environ.get("MAYA_GPUS", "0,1").split(",") if g.strip()]
    if not archs or min(archs) < 75:
        sys.exit(f"build: CUDA_ARCHS needs explicit architectures of 75 or newer (got {archs}); older ones "
                 "need the GPUs at build time, which a build container has not")
    return archs, gpus


def _compile_engine(target: str, exe, label: str) -> int:
    """The shared engine compile: the architectures and GPUs from the environment, llama.cpp's
    source, the nvcc and Maya's own source hash - then maya.compile_engine for the target.
    The stamp the last compile writes (build/MAYA-BUILD.json) is what the runtime setup reads:
    a start then reports the engine as already built and compiles nothing."""
    import setup as S     # noqa: PLC0415
    import maya           # noqa: PLC0415

    archs, gpu_ids = archs_and_gpus()
    llama = S.get_llama_cpp()
    nvcc, _ = maya.find_nvcc(archs)
    if nvcc is None:
        print("build: no CUDA toolkit (nvcc) found in the image", file=sys.stderr)
        return 1
    src = S.source_hash(S.ENGINE_SOURCES)
    meta = maya.compile_engine(archs, gpu_ids, nvcc, None, llama, src, target=target)
    if meta is None:
        print(f"build: the {label} engine build failed (the reason is above)", file=sys.stderr)
        return 1
    print(f"build: {label} built for {', '.join('sm_' + str(a) for a in meta['archs'])}: {exe}")
    return 0


def stage_engine() -> int:
    """The GLM engine (the `maya` target: GLM-5.3-Flash, GLM-5.3 and Qwen-3.8-Flash-Next)."""
    import maya           # noqa: PLC0415  the built engine's path

    return _compile_engine("maya", maya.EXE, "engine")


def stage_vision() -> int:
    """The vision encoder, from the engine layer's stamp: llama.cpp's mtmd with the model's
    vision tower. Maya treats an encoder that did not compile as "text only"; here the build
    fails instead, so a broken encoder never reaches a start quietly."""
    import setup as S     # noqa: PLC0415
    import maya           # noqa: PLC0415

    if not maya.STAMP.exists():
        print("build: the engine's stamp is missing - the engine stage runs first", file=sys.stderr)
        return 1
    meta = json.loads(maya.STAMP.read_text())
    pc = {"backend": "cuda", "archs": meta["archs"], "nvcc": meta["nvcc"],
          "gpus": [{"index": g} for g in meta.get("gpu_ids", [])]}
    src = S.source_hash(("tools/vision",))
    print(f"build: compiling the vision encoder for "
          f"{', '.join('sm_' + str(a) for a in meta['archs'])} ...")
    if not maya.compile_vision(pc, meta, Path(meta["llama"]), src):
        print("build: the vision encoder build failed (the reason is above)", file=sys.stderr)
        return 1
    print(f"build: vision encoder built: {maya.VEXE}")
    return 0


def stage_hy4() -> int:
    """The Hy4-Flash engine beside the GLM one: the same build folder and stamp keep both
    targets ("maya", "maya-hy4"), so only its own sources compile here - a start finds it
    built instead of compiling maya-hy4 at every container recreation."""
    import maya           # noqa: PLC0415  the built engine's path

    return _compile_engine("maya-hy4", maya.HY4_EXE, "maya-hy4")


STAGES = {"env": stage_env, "engine": stage_engine, "vision": stage_vision, "hy4": stage_hy4}


def ensure_venv(rest: list[str]) -> int:
    """Run with a plain python3: create the venv if it is not there yet and re-run this
    script with the venv's interpreter, so pip and the imports below land in the venv."""
    import venv as venv_mod     # noqa: PLC0415  stdlib; this early stage needs nothing else

    if not venv_python().exists():
        print(f"build: creating the Python environment in {ROOT / '.venv'} ...")
        venv_mod.create(str(ROOT / ".venv"), with_pip=True)
    if not in_venv():
        os.execv(str(venv_python()), [str(venv_python()), __file__, *rest])


def main() -> int:
    ap = argparse.ArgumentParser(description="Maya's image build setup: the .venv, llama.cpp's source, "
                                             "the engines (maya, maya-hy4) and the vision encoder, "
                                             "compiled at image build time")
    ap.add_argument("--stage", choices=["all", *STAGES], default="all",
                    help="env (the .venv and llama.cpp), engine, vision or hy4; all runs them in order")
    args, rest = ap.parse_known_args()

    if not in_venv():
        # creates the venv when missing, then execv's into it - this call never returns
        ensure_venv(rest)

    stages = ["env", "engine", "vision", "hy4"] if args.stage == "all" else [args.stage]
    for stage in stages:
        rc = STAGES[stage]()
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
