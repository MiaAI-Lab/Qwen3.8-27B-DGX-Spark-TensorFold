# Shared settings for start.sh, stop.sh and scripts/*.sh. Any value can be overridden from the environment,
# e.g. `PORT=9000 ./start.sh` or `PULL=0 scripts/prepare.sh`, or set in ./.env: KEY=value lines, read here (never
# run as a script); a variable already set in the environment wins over the file. .env is yours, not the repository's.
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi

# The target and its DFlash2 drafter (CUDA serves the 27B with DFlash2 unless --no-drafts is given).
MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-27B-MLX-4bit}"           # MLX affine 4-bit, groups of 64 (~15 GB)
DRAFT_ID="${DRAFT_ID:-z-lab/Qwen3.8-27B-DFlash2}"             # the drafter (~3.6 GB); "" serves without drafts
# The patches (if any) and start.sh's flags are made for TensorFold v0.6.0 exactly (c464617). After changing
# TF_VERSION, TF_REPO or BASE_IMAGE, run `scripts/prepare.sh --rebuild`.
TF_VERSION="${TF_VERSION:-v0.6.0}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE="${IMAGE:-tensorfold-qwen38-27b:${TF_VERSION}}"          # the local image prepare.sh builds
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-27b-tf}"              # the server's container
# The prebuilt image: prepare.sh pulls $GHCR_IMAGE:<TF_VERSION>-<patches hash> before building (PULL=0 builds instead);
# scripts/publish-image.sh pushes it. Empty: always build locally.
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-27b-dgx-spark-tensorfold}"

SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B}"   # the model id clients see in /v1/models and replies (tensorfold --name)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"

# Serving defaults (./start.sh arguments come after them and win).
PARALLEL="${PARALLEL:-8}"          # requests decoded together (streams); 1 serves one at a time
# YaRN (patches/0002, opt-in): YARN_FACTOR=4 stretches the 262,144-token native window 4x (1,048,576 tokens). It rescales
# the rotary frequencies of the 16 attention layers, so it can cost a little quality at short range; leave it empty
# unless you need the length. A bigger window also needs memory: 64 KiB of KV a token (16 GiB per 262k tokens).
YARN_FACTOR="${YARN_FACTOR:-}"
if [[ -n "$YARN_FACTOR" ]]; then
  export TENSORFOLD_YARN_FACTOR="$YARN_FACTOR"
  _native=$(awk -v f="$YARN_FACTOR" 'BEGIN { printf "%d", 262144 * f }')
else
  _native=262144
fi
CONTEXT="${CONTEXT:-$_native}"       # prompt + reply window per stream (the model's native maximum, times YARN_FACTOR)
# bf16 prompt activations are the engine's default; FP8 (e4m3) prompts fill ~35-40% faster at a measurable cost in
# prompt precision (README "Prefill precision"). Decode is not affected.
PREFILL_FP8="${PREFILL_FP8:-1}"
# Image input (--vision) works on the MLX checkpoint only (TensorFold refuses it for nvidia/Qwen3.8-27B-NVFP4).
VISION="${VISION:-1}"      # 0: text only
VISION_URLS="${VISION_URLS:-0}"    # 1: also accept public https:// image URLs (default: data URLs only)
# Thinking mode (Qwen's recommended sampling): temperature 1.0, top_p 0.95, top_k 20. A request's own values win.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
THINKING="${THINKING:-1}"
CHECKPOINT_SLOTS="${CHECKPOINT_SLOTS:-}"   # retained prompt-end states (tensorfold --checkpoint-slots); empty: default

# Images and video (patches/0001): images a request may carry and the visual tokens they share (each image at most
# 4,096; 50 images get ~327 each), and a request's video token budget (2 frames a second, at most 256 frames).
export TENSORFOLD_MAX_IMAGES="${TENSORFOLD_MAX_IMAGES:-50}"
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"

# Attention cache precision (patches/0003): fp8 (e4m3, one byte a value, no scale) halves the KV cache, 64 -> 32 KiB a
# token, which doubles the tokens that fit; bf16 is TensorFold's own. Prompts and decode read the same rounded values,
# so drafted replies still equal serial ones.
KV_DTYPE="${KV_DTYPE:-fp8}"
export TENSORFOLD_KV_DTYPE="$KV_DTYPE"

# Pinned KV pool (patches/0005): GiB of attention cache reserved once at startup and shared by every stream and kept
# prompt state; the caches grow inside it, nothing is given back, and a request that does not fit waits. 32 KiB a
# token with the FP8 cache, so 78 GiB is ~2.5M tokens. Empty or 0: caches grow on demand (the 20 GiB you see at idle).
# The startup estimate counts it, so it must fit beside the weights and per-stream buffers (~37 GiB at 16 streams).
KV_POOL_GB="${KV_POOL_GB:-78}"
[[ -z "$KV_POOL_GB" || "$KV_POOL_GB" == 0 ]] || export TENSORFOLD_KV_POOL_GIB="$KV_POOL_GB"

# Startup reserve: GiB left out of MemAvailable at admission and kept free by the stream memory gate. TensorFold's own
# default is max(4 GiB, a tenth of RAM) and its floor 2; patches/0004 allows 0, which is the default here: the budget
# is all of MemAvailable (the page cache counts as available). On the Spark's unified memory, running out can freeze
# the machine instead of failing an allocation: raise this (e.g. 4) if other workloads share the box.
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-0}"
# No "is there a newer TensorFold" call to GitHub at each start. 0: check.
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# Persists compiled CUDA kernels (torch extensions + triton) so only the first start pays the compile.
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38-27b}"

MIN_FREE_GB="${MIN_FREE_GB:-25}"       # free disk the downloads need (the target ~15 GB + the drafter ~3.6 GB)
IMAGE_FREE_GB="${IMAGE_FREE_GB:-35}"   # free disk under Docker's root that pulling or building the image needs
MIN_AVAIL_GB="${MIN_AVAIL_GB:-60}"     # MemAvailable start.sh warns below (the server's own admission is the real check)

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

model_cache_dir() { echo "$HF_CACHE/hub/models--${1//\//--}"; }
have_model()      { ls -d "$(model_cache_dir "$1")"/snapshots/*/ >/dev/null 2>&1; }

# The patches baked into $IMAGE, in order. None yet is fine: the hash of nothing still labels the image.
patch_files() { ls patches/*.patch 2>/dev/null || true; }
patches_hash() { patch_files | xargs -r cat | sha256sum | cut -c1-12; }

# What scripts/prepare.sh last left ready (it writes this line to PREPARED_MARKER when it succeeds); start.sh runs
# prepare.sh again whenever the current line differs: a missing or stale image, new patches, another model.
PREPARED_MARKER="$KERNEL_CACHE/.prepared"
prepared_state() {
  local hash label model=missing draft=none
  hash=$(patches_hash)
  label=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || echo missing)
  have_model "$MODEL_ID" && model=present
  if [[ -n "$DRAFT_ID" ]]; then draft=missing; have_model "$DRAFT_ID" && draft=present; fi
  echo "model=$MODEL_ID($model) draft=$DRAFT_ID($draft) image=$IMAGE($label) patches=$hash"
}
