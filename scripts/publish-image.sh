#!/usr/bin/env bash
# Push the image scripts/prepare.sh built to GitHub Container Registry as $GHCR_IMAGE:<TensorFold version>-<patches hash>
# and :latest, labelled with this repository (so the package takes the repository's visibility and access).
# Needs a GitHub token with write:packages: `gh auth login -s write:packages`, or GHCR_TOKEN / GITHUB_TOKEN set.
# Usage: scripts/publish-image.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

[[ -n "$GHCR_IMAGE" ]] || die "GHCR_IMAGE is empty (see scripts/config.sh)"
REPO_URL="${REPO_URL:-https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing, run scripts/prepare.sh first"
hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")
[[ "$hash" == "$(patches_hash)" ]] || die "$IMAGE was built from other patches; run scripts/prepare.sh first"
tag="${TF_VERSION}-${hash}"

token="${GHCR_TOKEN:-${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}}"
[[ -n "$token" ]] || die "no GitHub token: run gh auth login -s write:packages, or set GHCR_TOKEN"
user="${GHCR_USER:-$(gh api user --jq .login 2>/dev/null || echo "${GHCR_IMAGE#ghcr.io/}" | cut -d/ -f1)}"

log "Labelling $IMAGE as $GHCR_IMAGE:$tag"
docker build -q -t "$GHCR_IMAGE:$tag" -t "$GHCR_IMAGE:latest" \
  --label org.opencontainers.image.source="$REPO_URL" \
  --label org.opencontainers.image.description="Qwen3.8-27B on one DGX Spark: TensorFold $TF_VERSION with patches $hash" \
  --label org.opencontainers.image.licenses="Apache-2.0" \
  - <<<"FROM $IMAGE" >/dev/null
log "Pushing $GHCR_IMAGE:$tag and :latest (~25 GB uncompressed; only changed layers upload)"
# log in with a throwaway Docker config, so no registry credential stays in ~/.docker after the push
login_dir=$(mktemp -d)
trap 'rm -rf -- "$login_dir"' EXIT
echo "$token" | DOCKER_CONFIG="$login_dir" docker login ghcr.io -u "$user" --password-stdin >/dev/null
DOCKER_CONFIG="$login_dir" docker push "$GHCR_IMAGE:$tag"
DOCKER_CONFIG="$login_dir" docker push "$GHCR_IMAGE:latest"
log "Done: docker pull $GHCR_IMAGE:$tag"
