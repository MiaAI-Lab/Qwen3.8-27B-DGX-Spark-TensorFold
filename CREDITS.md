# Credits

This repository is a layer of scripts and patches. Almost everything that makes it work was built by others.

## Model and checkpoints

- **[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** by [Qwen](https://qwen.ai/): the model's design, training
  and evaluations. Released under the **Qwen Community License 1.0**, which governs any use of the weights (read it
  before commercial use). The weights are not part of this repository; `scripts/prepare.sh` downloads them.
- **[Vontra](https://huggingface.co/Vontra)**: the checkpoint served by default,
  [`Vontra/Qwen3.8-27B-MLX-4bit`](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit), the MLX 4-bit conversion
  (affine, groups of 64) that includes the vision tower.
- **[z-lab](https://huggingface.co/z-lab)**: the DFlash2 drafter,
  [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), a block-diffusion drafter that
  proposes the tokens the target then verifies.
- **[NVIDIA](https://huggingface.co/nvidia)**: [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4),
  the NVFP4 export measured as the comparison arm (`MODEL_ID=nvidia/Qwen3.8-27B-NVFP4` serves it; no vision).

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors. v0.6.0 is Apache-2.0 (releases through v0.5.0 stay MIT). The engine does all the serving:
  the Qwen3.8 dense CUDA engine (lane matmuls, DeltaNet and attention kernels, FP8 prompt path), DFlash2 draft trees
  and context copies, the concurrent scheduler and memory gate, the OpenAI-compatible server and the Qwen image
  pipeline. **Every file in `patches/` is a modification of TensorFold v0.6.0** (`c464617`); see `LICENSE`, `NOTICE`
  and `LICENSES/`.
- TensorFold itself builds on, and credits in its
  [third-party notices](https://github.com/ashhart/TensorFold/blob/v0.6.0/THIRD_PARTY_NOTICES.md):
  [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple, MIT),
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (Prince Canuma, MIT),
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT) and Hugging Face
  [transformers](https://github.com/huggingface/transformers) (Apache 2.0).

## The recipe this is adapted from

- **[MiaAI-Lab](https://github.com/MiaAI-Lab)** ([x.com/MiaAI_lab](https://x.com/MiaAI_lab)), the
  [Qwen3.8 Flash Next on one DGX Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
  recipe (MIT, kept under its notice in `LICENSES/MIT-MiaAI-Lab.txt`). `start.sh`, `stop.sh`, `scripts/prepare.sh`, `scripts/config.sh`, `tools/bench.py`, `tools/needle.py`,
  `tools/toolcheck.py`, `tools/visioncheck.py` and the layout and measurement method of this README are adapted from it.
  The **video** support in `patches/0001` (`vision/videos.py`, the timestamped frame-group prompt, the frame encoding)
  is ported from that recipe's `patches/0002-flash-next-v060.patch` (MIT); MiaAI-Lab's work there builds on TensorFold's Qwen
  image pipeline, Hugging Face transformers' Qwen3-VL video processing, and PyAV.
- MiaAI-Lab's [Qwen3.8-27B SGLang recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) and
  [GLM-5.3-Flash EXL3 two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) were
  read as references (per-token KV cost, a pool sized at startup, FP8 KV and YaRN settings). **No code was taken from
  them.** The 50-image and 16,384-shared-token figures follow the ones MiaAI-Lab's Flash Next and GLM READMEs state; the
  code that enforces them in 0001 is new.

## What this repository adds (`patches/`)

- **0001**: up to 50 images a request (limits, groups of up to 16,384 patches per tower call) and the port of video
  input, on TensorFold's Qwen vision frontend.
- **0002**: opt-in YaRN rope scaling for the dense engine. The frequency blend and the attention scale follow the
  YaRN method ([Peng, Quesnelle, Fan, Shippole, *YaRN: Efficient Context Window Extension of Large Language Models*,
  2023](https://arxiv.org/abs/2309.00071)) as implemented in Hugging Face transformers' `_compute_yarn_parameters`
  (Apache 2.0), which the patch is checked against numerically.
- **0003**: an FP8 (e4m3) attention cache read directly by the tree-attention kernels; **0004**: a memory reserve of 0;
  **0005**: a pinned KV pool for the concurrent decoder.
- Patches 0002-0005, the quality tools (`tools/kvquality.py`, `tools/qualitycheck.py`), `tools/mediacheck.py`,
  `tools/prefill_curve.py`, `scripts/banner.sh`, `docs/measurements.md` and this documentation were developed with
  [Claude Code](https://claude.com/claude-code).

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the base of the image, with NVIDIA's CUDA, cuDNN, cuBLAS and related libraries.
  Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for NVIDIA AI Products; see the
  README's License section.
- **[PyTorch](https://pytorch.org/)** (BSD-3-Clause), **[Triton](https://github.com/triton-lang/triton)** (MIT; most
  of the attention kernels are written in it) and **[NumPy](https://numpy.org/)** (BSD-3-Clause).
- **[Hugging Face transformers](https://github.com/huggingface/transformers)** (Apache 2.0): the Qwen vision tower's
  modules and the image processor; **[Hugging Face Hub](https://huggingface.co/)** and `huggingface_hub` (Apache 2.0):
  model hosting and the `hf` CLI.
- **[PyAV](https://github.com/PyAV-Org/PyAV)** (BSD-3-Clause) with its **[FFmpeg](https://ffmpeg.org/)** libraries
  (LGPL): video decoding. **[Pillow](https://python-pillow.org/)** (MIT-CMU): image decoding.
- **[Docker](https://www.docker.com/)** and the **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)**
  (Apache 2.0).

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10, 128 GB unified memory):
  every number in this repository was measured on one.

If you believe something here is missing or credited wrongly, please open an issue.
