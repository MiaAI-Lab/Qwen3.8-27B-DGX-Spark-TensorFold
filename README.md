<h1 align="center">Qwen3.8-27B on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

<p align="center">
  <img src=".github/image.png" alt="Qwen3.8 27B on TensorFold, single DGX Spark" width="100%" />
</p>

Serve **Qwen3.8-27B** from a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API, with up to
**8 concurrent requests**, a **pinned 78 GiB KV pool (2.5M tokens)**, the full **262,144-token context**, DFlash2 speculative decoding, **up to 50 images and video input**. It runs
[TensorFold](https://github.com/ashhart/TensorFold) v0.6.0 (`c464617`) in NVIDIA's PyTorch container, plus five patches
(`0001`: up to 50 images and video input; `0002`: opt-in YaRN, up to a 1,048,576-token window; `0003`: FP8 attention
cache, on by default; `0004`: a memory reserve of 0; `0005`: a pinned KV pool). The engine is upstream's.

- Checkpoint: [`Vontra/Qwen3.8-27B-MLX-4bit`](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit) (affine 4-bit, groups of 64, ~15 GB)
- Drafter: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (~3.6 GB)
- API model id: `Qwen3.8-27B`
- Images and videos in chat messages (`image_url` / `video_url` parts), see [Images and video](#images-and-video)
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## Performance

The official numbers, from [sparkDash](https://github.com/MiaAI-Lab/sparkDash), the benchmark of the MiaAI-Lab Spark
recipes (one DGX Spark; `agg` is the total across the concurrent requests, `str` the rate of each one, TTFT the time to
the first token).

**Decode, prose** (tok/s)

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 58.3 | 58.3 | 100 ms |
| 2 | 108.5 | 55.4 | 208 ms |
| 4 | 135.1 | 40.3 | 417 ms |
| 8 | 220.6 | 32.7 | 870 ms |

**Decode, code** (tok/s)

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 136.9 | 136.9 | 101 ms |
| 2 | 223.9 | 119.6 | 202 ms |
| 4 | 292.0 | 89.2 | 427 ms |
| 8 | 323.2 | 53.3 | 867 ms |

**Prefill** (tok/s)

| Prompt | Tokens | Prefill speed | Time to first token |
| ---: | ---: | ---: | ---: |
| 8k | 8,231 | 1,793.0 tok/s | 4.59 s |
| 16k | 16,420 | 1,785.3 tok/s | 9.20 s |
| 32k | 32,802 | 1,633.3 tok/s | 20.08 s |
| 64k | 65,576 | 1,415.9 tok/s | 46.31 s |
| 128k | 131,111 | 1,117.8 tok/s | 117.29 s |

Longer prompts, measured here with a needle-in-a-haystack prompt on the default server (window 262,144): 195k tokens in 205 s
(948 tok/s) and 255,897 tokens in 315 s (812 tok/s). With YaRN: 491k in 924 s and 884k in 2,664 s
([guide](#longer-context-with-yarn-a-1m-token-window)).

**Our own run** with a fixed-seed benchmark script (2026-10-01, thinking off, 8 streams; other prompts and counting than
sparkDash, so the decode figures are lower and not comparable one to one): per request 48.6 tok/s alone, aggregate 93 / 136 / 150 tok/s at
2 / 4 / 8 clients; prefill 1,862 tok/s at 8k, 1,663 at 31k, 1,147 at 126k. `PARALLEL=16` reached 207 tok/s at 16
clients. A single request decodes the same at every `PARALLEL`.

## Pinned KV pool

`KV_POOL_GB` (patch 0005, default `auto`) reserves **up to 78 GiB of attention cache at startup**, 2,555,904 tokens at
32 KiB, and keeps it. `auto` sizes it from the memory free when `start.sh` runs (free GiB minus ~31 for everything else,
at most 78: a Spark with ~109 GiB free gets the full 78, one with 90 GiB free gets 59) and prints the result; a number
sets it, `0` turns the pin off. It reserves it and keeps it: the server process holds ~97 GiB from the first second (weights, drafter and buffers ~19 GiB, plus the pool) and
that number does not move. The streams' caches and the kept prompt states grow inside the pool, the memory gate counts
them against it, and a request that does not fit waits for others to finish. Nothing is given back to the system, and
the host's free memory is no longer consulted for KV, so other workloads cannot take it. Eight full 262,144-token
windows (2.1M tokens) fit in it at once.

- **How:** after startup the engine allocates one block of the pool's size, touches every page so it is really
  committed, and frees it into PyTorch's allocator without releasing it; the decoder's cache-trimming calls are
  disabled. The startup estimate counts the pool, so it must fit beside the fixed memory: 107.7 GiB of a 109.2 GiB
  budget at 8 streams, which is why **16 streams reach ~72 GiB** (`KV_POOL_GB=72 PARALLEL=16`, 2.36M tokens).
- **Measured:** 8 simultaneous ~59k-token prompts: 8 of 8 ok, process memory 96,693-96,715 MiB throughout, lowest
  `MemAvailable` 10.4 GiB. At 4 GiB free (`MemAvailable` right after start with the default settings) this is tight on a
  shared machine: size `KV_POOL_GB` down, or raise `TENSORFOLD_MEMORY_RESERVE_GIB`, if other containers need room.
- **Not a speed feature:** it avoids growth copies and keeps memory predictable; decode and prefill are unchanged.
- The pool counts everything torch holds beyond the startup baseline (the prompt kernel's transient widened copies,
  kept DeltaNet states), so a long prefill briefly uses some of it. One GPU only; with `KV_POOL_GB` empty the old
  grow-on-demand behavior is unchanged.

## FP8 KV cache

`KV_DTYPE=fp8` (the default; patch 0003) stores the 16 attention layers' keys and values as e4m3, one byte a value,
no scale (saturating at +-448): **32 KiB a token instead of 64**, so twice the tokens fit. `KV_DTYPE=bf16` is
TensorFold's own cache. The rows a forward attends to are rounded through e4m3 before use, so a key reads alike in a
draft window and in the cache and a drafted reply still equals the serial one (checked: greedy and sampled, text and
video, 6 of 6 token hashes equal). The tree-attention kernels read the FP8 cache directly; the prompt kernel is handed
each layer's rows widened to bf16 (a transient 2 x window x 2 KiB per layer, counted in the startup estimate).

| | bf16 cache | FP8 cache |
| --- | ---: | ---: |
| KV a token / a full 262,144-token stream | 64 KiB / 16 GiB | 32 KiB / 8 GiB |
| Startup estimate, 8 streams x 262,144 | 45.4 GiB | 37.8 GiB |
| KV pool at the admission budget (~107 GiB) | ~62 GiB, ~1.0M tokens | ~69 GiB, ~2.2M tokens |
| `YARN_FACTOR=4`, 1,048,576-token windows | 2 streams: 93.8 GiB | 4 streams: 66.1 GiB |
| Prefill 8k / 126k | 1,860 / 1,149 tok/s | 1,862 / 1,147 tok/s |
| C=8 aggregate decode | 150 tok/s | 146 tok/s |

Quality, measured in the engine on 8 sequences of 4,096 tokens (4 wikitext-2 test, 4 CPython source, two 2,048-row
chunks so the second reads cached keys), against the bf16 cache with bf16 prompts:

| | Perplexity | KL to reference | Top-1 agreement |
| --- | ---: | ---: | ---: |
| reference (bf16 cache, bf16 prompts) | 3.316 | | |
| **FP8 cache**, bf16 prompts | 3.315 | 0.0031 | 98.7% |
| bf16 cache, FP8 prompts (`PREFILL_FP8=1`) | 3.397 | 0.0435 | 94.1% |
| FP8 cache, FP8 prompts (the defaults) | 3.387 | 0.0419 | 94.0% |

The cache costs far less than the prompt precision: FP8 prompts are ~14x the KL of the FP8 cache, so if quality
matters more than prefill speed, `PREFILL_FP8=0` is the setting to change first. Greedy open-ended replies do
diverge from the bf16 cache's after a few tokens (as they do for any numeric change); a 12-question reasoning probe
scored 12/12 with the FP8 cache and 11/12 with bf16 (one retrieval miss: noise at this
size), and the needle is found at 195k and, with YaRN, at 491k tokens. Only 4k-token contexts were scored by KL;
long-context quality was checked with needles only.

## Longer context with YaRN: a 1M-token window

The model's native window is 262,144 tokens, and that is what the server runs by default: a longer prompt is refused
(a prompt plus its reply must fit the window). **YaRN** (patch 0002) stretches the window by a factor; `4` gives
**1,048,576 tokens**. It rescales the rotary frequencies of the 16 full-attention layers (frequencies and the
0.1 ln(f) + 1 attention scale as in transformers, checked against `_compute_yarn_parameters`: 1e-7 relative
difference). Any factor of 1 or more works.

**Enable 1M context**

```bash
YARN_FACTOR=4 ./start.sh restart
```

That is all: `CONTEXT` follows the factor (262,144 x 4 = 1,048,576), so nothing else needs setting. Make it permanent
by putting the lines in a `.env` file next to `start.sh`:

```bash
echo 'YARN_FACTOR=4' >> .env
./start.sh restart
```

Tested with exactly these settings: a 353,543-token prompt (above the native window) was prefilled in 533 s and the
needle in it found.

Check it took: the startup log must say `native 1048576, allocated prompt/reply window 1048576` and
`... pinned 70.0 GiB cache pool (2,293,760 tokens)`, `docker logs qwen38-27b-tf | grep -E 'estimate|pool|context'`, and
`curl -s localhost:8888/v1/models` answers. To go back: remove the two lines (or `YARN_FACTOR= ./start.sh restart`).

**The pool shrinks for 1M:** the startup estimate counts the pinned pool beside per-stream scratch that grows with the
window, and a 1M window needs ~9 GiB more of it than a 262k one, so `auto` picks ~70 GiB instead of 78 (a fixed
`KV_POOL_GB=78` is refused with a 1M window: the refusal names the window that would fit, 447,487 tokens; 70 starts, estimate
108.7 of 109.7 GiB). Tested settings:

| Setting (8 streams) | Result |
| --- | --- |
| `YARN_FACTOR=4` (`auto` pool: ~70) | starts: estimate 108.72 of 109.71 GiB, pool 2,293,760 tokens (two full 1M streams, or 8 streams of ~290k) |
| `YARN_FACTOR=4` with the default pool (78) | refused at startup, before any weights load: largest fitting window 447,487 tokens |

Fewer streams or `KV_POOL_GB=0` (no pin, caches grow on demand) leave more room; if a setting is refused, the message
names the window that would fit, so lower `KV_POOL_GB` or `PARALLEL` and try again.

A full 1M-token stream is 32 GiB of KV (FP8 cache; 64 GiB with `KV_DTYPE=bf16`).

**What to expect**
- **Prefill is slow at length** (attention grows with it): cold prefill measured at 205 s for 195k, 924 s for 491k and
  2,664 s (44 minutes) for 884k tokens. Send long prompts with a client timeout of an hour or more.
- **Quality:** YaRN rescales every position, so short-range replies change slightly (greedy text differs from the
  unscaled server's). Quality was checked with needles (195k, 491k, 884k tokens found), not beyond. Unset, the engine is
  byte-identical to before. If most of your prompts are short, leave YaRN off.
- The DFlash2 drafter uses a sliding window with relative positions, so it is not scaled; drafts are still verified
  against the model's own samples.
- A checkpoint whose own `config.json` says `rope_type: yarn` is read the same way; `YARN_FACTOR` overrides it.

## Images and video

TensorFold's own image support takes 4 images a request and no video. Patch 0001 raises that and adds video (the
video code is ported from the Flash Next recipe's patch). Send them as OpenAI-style content parts in a user message:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 2000
}'
```

A video is a `video_url` part (`{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`).

| | Images | Videos |
| --- | --- | --- |
| Formats | JPEG, PNG, WebP | MP4, WebM, MOV, MKV (anything FFmpeg decodes) |
| Per request | up to 50 (`TENSORFOLD_MAX_IMAGES`; all of a chat's turns count), 10 MB each, 64 MB in all | up to 2, 64 MB each, 96 MB in all, up to an hour of footage |
| Tokens | up to 16,384 for all images (`TENSORFOLD_IMAGE_TOKENS`), at most 4,096 an image (50 images: ~327 each; `"detail": "low"`: 256 an image) | 2 frames a second (at most 256 frames, spread over the whole video), each pair of frames one timestamped block; up to 16,384 tokens a request (`TENSORFOLD_VIDEO_TOKENS`) |

By default only data URLs are accepted; `VISION_URLS=1` also lets the server fetch public `https://` URLs. A request
body can be up to 96 MiB (base64 makes data URLs a third larger than the files). Image and video prompts are not
kept for prefix reuse, so each turn of a chat with media processes it again. Text requests are unaffected. Needs the
MLX checkpoint: the NVFP4 one cannot take images or video. Tested: 50 images in one request (a 51st is refused),
50 full-HD photos (15,735 tokens, 13 s), a 6-second video and a 90-second 1280x720 video (16,135 tokens, 12.5 s).

## What was tuned, and what was not

Tested on this box against TensorFold's defaults:

- **`--prefill-fp8` on** (`PREFILL_FP8=1`): +52% prefill (NVFP4 test, 12.6k: 1,848 vs 1,215 tok/s); upstream measures
  +35-40% on this checkpoint. It lowers prompt precision (upstream: KL 0.0624 vs 0.0031 against an fp32 reference,
  perplexity +1.3% wikitext / +4.1% code, for the MLX checkpoint). `PREFILL_FP8=0` gives bf16 prompts. Decode is
  unaffected, and drafted replies still equal serial ones.
- **`PARALLEL=8`** instead of upstream's one-at-a-time default: no cost for a single request, aggregate decode 3x
  (150 tok/s at 8 clients). `PARALLEL=16` also works (207 tok/s at 16 clients, ~8 GiB more fixed memory, so the pool
  tops out near 72 GiB).
- **`TENSORFOLD_MEMORY_RESERVE_GIB=0`** (patch 0004): TensorFold's budget is MemAvailable minus a reserve (a tenth of
  RAM by default, 2 GiB at least). Here it is all of MemAvailable: 109.4 GiB at the last start instead of 107.4.
- **MLX 4-bit over NVIDIA NVFP4**: NVFP4 prefilled the same but decoded 20-25% slower and cannot take images.
  `MODEL_ID=nvidia/Qwen3.8-27B-NVFP4 VISION=0 ./start.sh restart` serves it. Quality between the two was **not**
  measured here.

Not changed because nothing indicated a gain: prompt chunks (upstream already sizes them to 4,096 rows on this card),
draft tree width (upstream picks it per round from a measured cost curve), and the kernels (the 12k prefill is already
~100 TFLOPS). The Flash Next recipe's patches (copy drafts, SSD read-ahead, MTP settings) address that model's
engine and have no counterpart here. KV-cache quantization is Flash Next only.

## Requirements

- A DGX Spark (or another GB10 system with 128 GB unified memory). Startup estimate 45 GiB at the defaults.
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~60 GB free disk: ~19 GB for the checkpoints under `~/.cache/huggingface`, ~25-35 GB for the image.
- Optional: the `hf` CLI on the host (faster, resumable downloads).

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold.git
cd Qwen3.8-27B-DGX-Spark-TensorFold
./start.sh
```

The first run pulls the prebuilt image (`ghcr.io/miaai-lab/qwen3.8-27b-dgx-spark-tensorfold:v0.6.0-<patches hash>`, if the
package is reachable with your Docker login; else it builds it locally, a few minutes) and downloads the checkpoints, then
compiles the CUDA kernels (a few minutes, once).
Later starts take about a minute. `start.sh` runs a smoke test and prints the endpoint.

```bash
curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'

./start.sh restart                # apply changed settings
./stop.sh                         # stop and free the GPU memory
docker logs -f qwen38-27b-tf      # server log
```

The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`. Images go in as
`image_url` parts (data URLs; `VISION_URLS=1` also allows `https://`). Per request: `temperature`, `top_p`, `top_k`,
`seed`, `"chat_template_kwargs": {"enable_thinking": false}`, and `"draft": false` for the serial reference.

## Configuration

Every setting is in [`scripts/config.sh`](scripts/config.sh); override from the environment, a `.env` file next to
`start.sh`, or `tensorfold serve` flags after `start.sh` (`./start.sh restart --context 131072`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `8` | requests decoded together; `1` serves one at a time |
| `CONTEXT` | `262144` | prompt + reply window per stream (times `YARN_FACTOR`) |
| `YARN_FACTOR` | unset | YaRN rope scaling factor, e.g. `4` for a 1,048,576-token window ([guide](#longer-context-with-yarn-a-1m-token-window)) |
| `KV_POOL_GB` | `auto` | GiB of attention cache pinned at startup and shared by all streams: `auto` = the free memory minus ~31, at most 78 ([KV pool](#pinned-kv-pool)); a number sets it; `0`: grow on demand |
| `KV_DTYPE` | `fp8` | attention cache: `fp8` (e4m3, 32 KiB a token) or `bf16` (64 KiB) ([above](#fp8-kv-cache)) |
| `PREFILL_FP8` | `1` | FP8 prompt activations: faster prefill, lower prompt precision; `0` for bf16 |
| `VISION` / `VISION_URLS` | `1` / `0` | image and video input (MLX checkpoint only) / also fetch `https://` URLs |
| `TENSORFOLD_MAX_IMAGES` / `TENSORFOLD_IMAGE_TOKENS` | `50` / `16384` | images a request may carry and the tokens they share, each at most 4,096 |
| `TENSORFOLD_VIDEO_TOKENS` | `16384` | a request's video token budget |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | default sampling (Qwen's thinking-mode values) |
| `THINKING` | `1` | open a think block by default |
| `CHECKPOINT_SLOTS` | unset | retained prompt-end states for prefix reuse (TensorFold default) |
| `MODEL_ID` / `DRAFT_ID` | MLX 4-bit / DFlash2 | other checkpoint; `DRAFT_ID=` serves without drafts |
| `SERVED_NAME`, `PORT`, `HOST`, `CONTAINER_NAME`, `IMAGE` | see file | |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `0` | GiB left out of MemAvailable at admission and kept free by the stream memory gate (patch 0004 allows 0; TensorFold's own default is a tenth of RAM, its floor 2). On this machine's unified memory, running out can freeze the host: raise it if other workloads share the box |

## Checks

`start.sh` ends with a smoke test, and `/health` shows the busy flag and live token totals. Every figure in this README
came from an OpenAI-compatible client talking to `/v1/chat/completions` (streaming, fixed seeds, `return_token_ids`
for exactness checks), so any such client reproduces them. The benchmark and check scripts used for the measurements
here (decode and prefill benchmark, needle, tool call, image and video, KL and quality probes) are not part of this
repository.

## Repository layout

```
start.sh, stop.sh   set up (first run) and start / stop the server
scripts/            prepare.sh (image + checkpoints), config.sh (all settings), banner.sh (start.sh's banner),
                    publish-image.sh (push the image to GitHub Container Registry)
patches/            patches baked into the image (0001: 50 images and video; 0002: YaRN; 0003: FP8 KV cache; 0004: memory reserve 0; 0005: KV pool)
LICENSE, NOTICE, LICENSES/, CREDITS.md   Apache-2.0 license, notices of the MIT parts, and who built what
```

## License

Apache License 2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The patches modify TensorFold v0.6.0, which is
Apache-2.0. Parts of the scripts and tools are adapted from MiaAI-Lab's MIT-licensed Flash Next recipe and keep its
notice ([`LICENSES/MIT-MiaAI-Lab.txt`](LICENSES/MIT-MiaAI-Lab.txt)); TensorFold's pre-0.6.0 MIT notice is in
`LICENSES/` too. The model weights, downloaded from Hugging Face and not part of this repository, are under the Qwen
Community License 1.0. The image is based on NVIDIA's PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`), governed by
NVIDIA's software license terms, which the container prints at every start; by pulling or running it you accept them.
It also contains Hugging Face `transformers` (Apache 2.0) and PyAV (BSD) with its FFmpeg libraries (LGPL).

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (v0.6.0), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
by Qwen, [Vontra's MLX 4-bit checkpoint](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit) and
[z-lab's DFlash2 drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), with the launcher, tools and the video code
adapted from MiaAI-Lab's Flash Next recipe. The full list, including the YaRN source, the runtime stack and licenses,
is in [`CREDITS.md`](CREDITS.md).
