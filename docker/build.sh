#!/usr/bin/env bash
# Build a container image of Project Maya from THIS repository, check it, and optionally push it. docs/DOCKER.md
# has the details. Run it from a checkout of this repository:
#
#   docker/build.sh                          # this checkout's HEAD (the committed state), CUDA 13, this PC's cards
#   docker/build.sh --ref v1.1.51            # another ref (any tag or commit of this repository)
#   docker/build.sh --cuda 12                # CUDA 12 base (12.9.1) instead of 13 (13.0.0)
#   docker/build.sh --archs "86;89"          # other CUDA architectures (default 86;120: the RTX 3090 + RTX 5090)
#   docker/build.sh --image ghcr.io/<you>/maya --push
#   docker/build.sh --check-only <image>     # run the checks on an image built before
#
# Options:
#   --ref REF          what to build (default: HEAD). Built from `git archive REF`, so the working tree's dirt and
#                      untracked files never reach the image - the image carries exactly a committed state, and the
#                      commit it was built from is written into .maya-image-ref (every start logs it)
#   --cuda 13|12       the CUDA base (default 13). 13: CUDA 13.0 (Blackwell needs 12.8+), driver 580 or newer.
#                      12: CUDA 12.9.1, driver 575 or newer
#   --archs LIST       the architectures the engines are compiled for (a start on cards outside the list
#                      recompiles into the container: nvcc and the sources are in the image)
#   --gpus LIST        MAYA_GPUS, recorded in the build stamp (default 0,1)
#   --image NAME       the image name (default project-maya); tagged NAME:<version>-cuda<13|12>
#   --push             push the tag (NAME must include a registry you can push to)
#   --check-only IMG   skip the build and check IMG (give --cuda / --archs / --ref as it was built)
#
# Every check runs without a GPU, and a failure stops the run before anything is pushed:
#   - VERSION is the release (when REF is a vX.Y.Z tag);
#   - setup.py's llama.cpp pin is the one CMakeLists.txt names (when it names one);
#   - the image carries the local source: the ablation kernels (src/core/glm_ablate.cuh) and the deferred start
#     (MAYA_START_UNLOADED in serve/server.py) - an image of a tree without the local commits never ships;
#   - .maya-image-ref is there and names the commit;
#   - the engines were built: build/maya and build/maya-hy4, and the vision encoder build-vision/bin/maya-vision;
#   - every library of maya and maya-hy4 resolves, except the host driver's (libcuda, libnvidia-*);
#   - serve/server.py and the three pack builders start (--help);
#   - the expert profiles and MiniMax's chat template are there;
#   - gguf-py imports the way the tools find it, and tools/iq_pack.py indexes data/glm5-synth.gguf;
#   - every package in setup.py's PY_PACKAGES (less cmake and ninja) is installed at a version it accepts.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

REF="" CUDA=13 ARCHS="" GPUS="0,1" IMAGE="project-maya" PUSH=0 CHECK_ONLY=""
usage() { awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="${2:?--ref needs a value}"; shift 2 ;;
    --cuda) CUDA="${2:?--cuda needs 13 or 12}"; shift 2 ;;
    --archs) ARCHS="${2:?--archs needs a list}"; shift 2 ;;
    --gpus) GPUS="${2:?--gpus needs a list}"; shift 2 ;;
    --image) IMAGE="${2:?--image needs a name}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --check-only) CHECK_ONLY="${2:?--check-only needs an image}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[maya-docker] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

say() { echo "[maya-docker] $*"; }
die() { echo "[maya-docker] ERROR: $*" >&2; exit 1; }

case "$CUDA" in
  13) CUDA_VERSION=13.0.0 ;;
  12) CUDA_VERSION=12.9.1 ;;
  *) die "--cuda takes 13 or 12, not '$CUDA'" ;;
esac
ARCHS="${ARCHS:-86;120}"
[[ "$ARCHS" =~ ^[0-9]+(;[0-9]+)*$ ]] || die "--archs takes numbers separated by ';', e.g. \"86;120\", not '$ARCHS'"
command -v docker >/dev/null || die "docker is not installed"

root="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)" \
  || die "run this from a git checkout of Project Maya"
REF="${REF:-HEAD}"
commit="$(git -C "$root" rev-parse -q --verify "$REF^{commit}")" || die "'$REF' is not a tag or commit of this repository"
version="$(git -C "$root" show "$commit:VERSION" | tr -d '[:space:]')"
[[ -n "$version" ]] || die "VERSION is empty at $REF"
if [[ "$REF" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  [[ "$version" == "${REF#v}" ]] || die "VERSION at $REF says '$version', not ${REF#v}"
fi
tag="$IMAGE:$version-cuda$CUDA"
ref_name="local"; [[ "$REF" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] && ref_name="$REF"

# ── checks on a built image ─────────────────────────────────────────────────────────────────────────────────────
check_image() {
  local ref="$1" got missing out e
  local run=(docker run --rm --entrypoint)

  got="$("${run[@]}" cat "$ref" /opt/project-maya/.maya-image-ref 2>/dev/null)" \
    || die "$ref has no /opt/project-maya/.maya-image-ref: not built by this recipe"
  [[ -n "$got" ]] || die "$ref's .maya-image-ref is empty"
  say "check: the image names what it was built from: $got"

  got="$("${run[@]}" cat "$ref" /opt/project-maya/VERSION 2>/dev/null | tr -d '[:space:]')" || die "$ref has no /opt/project-maya/VERSION"
  [[ "$got" == "$version" ]] || die "/opt/project-maya/VERSION says '$got', not $version"
  say "check: VERSION $version"

  # the local source is in the tree: an image of an un-patched Maya never ships
  "${run[@]}" test "$ref" -f /opt/project-maya/src/core/glm_ablate.cuh || die "$ref carries no ablation kernels (src/core/glm_ablate.cuh)"
  "${run[@]}" grep -q MAYA_START_UNLOADED /opt/project-maya/serve/server.py || die "$ref's serve/server.py has no MAYA_START_UNLOADED"
  say "check: the local commits are in the tree (the ablation kernels, the deferred start)"

  for e in maya maya-hy4; do
    "${run[@]}" test "$ref" -x "/opt/project-maya/build/$e" || die "$ref has no /opt/project-maya/build/$e"
    missing="$("${run[@]}" ldd "$ref" "/opt/project-maya/build/$e" | grep 'not found' | grep -v -E 'libcuda\.so|libnvidia' || true)"
    [[ -z "$missing" ]] || die "libraries of build/$e missing from the image:
$missing"
  done
  "${run[@]}" test "$ref" -x /opt/project-maya/build-vision/bin/maya-vision || die "$ref has no build-vision/bin/maya-vision"
  say "check: the engines (maya, maya-hy4) and the vision encoder are built; every library resolves except the host driver's"

  for e in serve/server.py tools/iq_pack.py tools/ds4/pack_gguf.py tools/mm3/pack_gguf.py; do
    out="$("${run[@]}" /opt/project-maya/.venv/bin/python "$ref" "/opt/project-maya/$e" --help 2>&1)" || die "$e --help failed:
$out"
  done
  for e in data/expert-profile-glm.bin data/expert-profile-qwen.bin data/expert-profile-ds4.bin data/expert-profile-mm3.bin \
           tools/mm3/chat_template.jinja; do
    "${run[@]}" test "$ref" -s "/opt/project-maya/$e" || die "$e is missing from the image"
  done
  say "check: serve/server.py and the three pack builders start; the expert profiles and MiniMax's chat template are there"

  # The release's 12 MB synthetic GLM-5 model has no tokenizer, so the pack builder's last step (the tokenizer)
  # can't run on it: the check wants the index and the dense weights.
  out="$("${run[@]}" sh "$ref" -c 'cd /opt/project-maya \
    && .venv/bin/python -c "import sys; sys.path.insert(0, \"tools\"); import _paths; _paths.add_gguf_py(); import gguf" \
    && { .venv/bin/python tools/iq_pack.py --gguf data/glm5-synth.gguf --out /tmp/pack --compat-bf16 >/tmp/pack.log 2>&1
         if test -s /tmp/pack/index.txt && test -s /tmp/pack/dense.bin; then echo ok; else tail -5 /tmp/pack.log; echo failed; fi; }' 2>&1)" || true
  [[ "$(tail -1 <<<"$out")" == ok ]] || die "gguf-py or the pack builder doesn't work in the image:
$out"
  say "check: gguf-py imports and tools/iq_pack.py indexes data/glm5-synth.gguf"

  # read from the image's own setup.py, so a package the release adds and the image lacks fails here
  out="$(docker run --rm -i --entrypoint /opt/project-maya/.venv/bin/python "$ref" - 2>&1 <<'PY'
import ast, sys
from importlib import metadata
from pip._vendor.packaging.requirements import Requirement
tree = ast.parse(open("/opt/project-maya/setup.py", encoding="utf-8").read())
found = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "PY_PACKAGES" for t in node.targets)]
if len(found) != 1:
    sys.exit("setup.py has no single PY_PACKAGES list")
bad, n = [], 0
for spec in ast.literal_eval(found[0]):
    req = Requirement(spec)
    if req.name.lower() in ("cmake", "ninja"):
        continue
    n += 1
    try:
        version = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        bad.append(f"{spec}: not installed")
        continue
    if req.specifier and not req.specifier.contains(version, prereleases=True):
        bad.append(f"{spec}: {version} installed")
print("\n".join(bad))
print("ok" if n and not bad else "failed")
PY
)" || true
  [[ "$(tail -1 <<<"$out")" == ok ]] || die "the image's .venv lacks packages setup.py installs:
$out"
  say "check: every package in setup.py's PY_PACKAGES (less cmake and ninja) is installed at a version it accepts"
}

if [[ -n "$CHECK_ONLY" ]]; then
  check_image "$CHECK_ONLY"
  say "$CHECK_ONLY passes every check"
  exit 0
fi

# ── the build ────────────────────────────────────────────────────────────────────────────────────────────────────
llama="$(git -C "$root" show "$commit:setup.py" | grep -m1 -E '^LLAMA_CPP_COMMIT *= *"[0-9a-f]{40}"' | grep -o -E '[0-9a-f]{40}' || true)"
[[ -n "$llama" ]] || die "setup.py at $REF has no LLAMA_CPP_COMMIT"
cmake_pin="$(git -C "$root" show "$commit:CMakeLists.txt" | grep -m1 -E 'GIT_TAG +[0-9a-f]{40}' | grep -o -E '[0-9a-f]{40}' || true)"
[[ -z "$cmake_pin" || "$cmake_pin" == "$llama" ]] \
  || die "setup.py pins llama.cpp $llama but CMakeLists.txt names $cmake_pin"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git -C "$root" archive "$commit" | tar -x -C "$work"
# the commit the image was built from, which every start logs (the answer to "which version runs here?")
printf '%s %s (%s)\n' "$ref_name" "${commit:0:12}" "$version" > "$work/.maya-image-ref"
say "building $tag from $REF (${commit:0:8}): CUDA $CUDA_VERSION, archs $ARCHS, llama.cpp ${llama:0:7}"
docker build --pull -f "$root/docker/Dockerfile" \
  --build-arg "CUDA_VERSION=$CUDA_VERSION" \
  --build-arg "CUDA_ARCHS=$ARCHS" \
  --build-arg "MAYA_GPUS=$GPUS" \
  --label "org.opencontainers.image.title=project-maya" \
  --label "org.opencontainers.image.description=Project Maya $version (the local build: the GLM ablation and the deferred start), CUDA $CUDA_VERSION, archs $ARCHS" \
  --label "org.opencontainers.image.source=https://github.com/mw00/project-maya" \
  --label "org.opencontainers.image.revision=$commit" \
  --label "org.opencontainers.image.version=$version" \
  --label "org.opencontainers.image.licenses=MIT" \
  -t "$tag" "$work"

check_image "$tag"

if [[ "$PUSH" != 1 ]]; then
  say "built and checked $tag (not pushed; --push publishes)"
  exit 0
fi
docker push -q "$tag" >/dev/null
digest="$(docker image inspect "$tag" --format '{{join .RepoDigests "\n"}}' | grep -m1 "^${IMAGE}@" || true)"
say "pushed $tag${digest:+ = $digest}"
