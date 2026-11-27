# syntax=docker/dockerfile:1
#
# A container image of Project Maya built from THIS repository: docker/build.sh drives it, taking the tree from
# `git archive` of the ref it is asked for (default: this checkout's HEAD), so the working tree's dirt never
# reaches the image and the image carries exactly a committed state. The commit it was built from is written into
# .maya-image-ref by build.sh before the build and is logged by every start - the answer to "which version runs
# here?". Everything a start would otherwise compile or download is built INTO the image here: Maya's Python
# environment (.venv), llama.cpp's source, the engines (maya, maya-hy4) and the vision encoder - all through
# Maya's own functions (docker/build.py), so a container start finds every stamp in place and compiles nothing.
# The pack beside the model files is the only thing a first start still builds (it indexes the 90 GB of GGUF
# files, which are mounted, not baked). What a start on OTHER GPUs than CUDA_ARCHS does: recompile into the
# container - hence the devel base stays in the image, not a slim runtime one. The container start itself is
# docker/start.py: the two models' configurations kept in /state and refreshed from MAYA_* at every start, the
# server (with MAYA_START_UNLOADED=1, no model loaded) in a session of its own.
#
# CUDA 13.0 builds for both this PC's cards (RTX 3090 = sm_86, RTX 5090 = sm_120; Maya needs 12.8+ for Blackwell)
# and runs against the installed 615.x driver.
#
# The compile goes through Maya's own setup functions (docker/build.py -> maya.compile_engine), whose cmake
# invocation ccache cannot be handed to the way an image built by a direct cmake call would pass it. ccache
# applies anyway in its other form: it masquerades as the compilers (first on the PATH) and keeps its cache in a
# BuildKit cache mount that outlives the build - a rebuild after a small source change recompiles only what the
# change touched.
ARG CUDA_VERSION=13.0.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

# the GPU architectures the engine is compiled for at build time; rebuild with another CUDA_ARCHS to move the
# image to other cards. MAYA_GPUS is only recorded in the build stamp
ARG CUDA_ARCHS=86;120
ARG MAYA_GPUS=0,1

# maya.sh needs Python 3.10+ with venv and ensurepip; the engine build needs g++/make (build-essential); ccache
# fronts the compilers (see the header); cmake and ninja come with Maya's own pip packages. The videos the server
# reads sent like pictures (v1.1.29's video support) need the ffmpeg/ffprobe pair on the PATH: the server checks
# for it, ffprobe times the video, the vision encoder reads its frames (MAYA_VIDEO_MAX_FRAMES, 4 a second)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        python3 \
        python3-venv \
        python3-pip \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ccache masquerades as the compilers it will be asked for: the engine's objects land in the cache mount of the
# build below, keyed on the source's and the flags' bits. CCACHE_COMPILERCHECK=content: the key holds the
# compiler's bits, not its (COPY-new) mtimes; CCACHE_PATH tells ccache where the real compilers live, so the
# lookup never loops back to the symlinks
ENV PATH="/usr/lib/ccache:${PATH}" \
    CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=30G \
    CCACHE_COMPILERCHECK=content \
    CCACHE_PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
# nvcc has no symlink in /usr/lib/ccache (CUDA is installed after ccache's package): add it
RUN ln -s /usr/bin/ccache /usr/lib/ccache/nvcc

# the tree as committed at the ref build.sh archived (no .git in an archive): the server (serve/server.py), the
# image build (docker/build.py), the start (docker/start.py), the pack builders
COPY . /opt/project-maya

# the heavy layer: Maya's Python environment, llama.cpp's source, the engines and the vision encoder - the one
# setup step build.py performs, in order (env, engine, vision, hy4). ccache's objects and pip's downloads live in
# cache mounts that outlive the build
RUN --mount=type=cache,id=project-maya-ccache,target=/root/.cache/ccache \
    --mount=type=cache,id=project-maya-pip,target=/root/.cache/pip \
    ccache --zero-stats >/dev/null; \
    CUDA_ARCHS="$CUDA_ARCHS" MAYA_GPUS="$MAYA_GPUS" python3 /opt/project-maya/docker/build.py; \
    rc=$?; ccache --show-stats; exit $rc

# the engine's measured data, seeded into the mounted data/ folder at every start: the image's part (the expert
# profiles and the seeds every load reads - the engine never measures them back) is copied aside here, so mounting
# an empty data/ folder loses nothing. The per-PC measurements the engines write (the prompt routes, the CPU lane
# timings) are NOT in the image: they land in the mount and survive a container recreate
RUN cp -a /opt/project-maya/data /opt/project-maya/.maya-data-image

# the image is the source of truth: never check for a newer Maya from inside the container (update by building a
# new ref: docker/build.sh, then docker compose up -d)
ENV MAYA_UPDATE_CHECK=0

WORKDIR /opt/project-maya
EXPOSE 8080

ENTRYPOINT ["python3", "/opt/project-maya/docker/start.py"]
