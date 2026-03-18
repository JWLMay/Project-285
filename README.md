# Low-latency Diffusion-based Image Dehazing and Risk Evaluation for Autonomous Systems

This repository contains our implementation and experiments for accelerating `DiffDehaze` toward deployment-oriented image dehazing for autonomous systems.

The work is documented in [Final_report.pdf](Final_report.pdf) and focuses on:

- low-latency diffusion dehazing through step reduction
- AccSamp-based accelerated inference
- INT8 quantization-aware training (QAT)
- safety-proxy evaluation for autonomous perception pipelines

The codebase builds on the original `Learning Hazing to Dehazing` / `DiffDehaze` implementation and extends it with profiling, evaluation, and QAT-oriented inference/training utilities.

## Project Summary

Diffusion dehazing models produce strong restoration quality, but their iterative reverse process is expensive for real-time systems. This project studies how far `DiffDehaze` can be accelerated without making the outputs unsuitable for safety-critical perception.

The report introduces two main acceleration directions:

1. Step reduction for reverse diffusion
2. Quantization-aware training for reduced-precision deployment

To evaluate whether faster dehazing remains usable for autonomous systems, the project also introduces a safety-proxy evaluation framework based on image-level Safety Performance Indicators (SPIs).

From the final report:

- up to `4.2x` inference speedup
- only `1.9%` degradation in SPI metrics
- `20-step` inference gives a `2.2x` speedup with only a small SPI drop relative to the vanilla setting

## What Is In This Repo

Core model and sampler code:

- `diffbir/`: diffusion model, ControlLDM, samplers, utilities
- `datasets.py`: stage-1 and stage-2 dataset definitions
- `configs/`: train and inference configs

Training and inference entry points:

- `train_stage1.py`: train haze generation stage
- `train_stage2.py`: train dehazing stage
- `train_stage2_QAT.py`: stage-2 quantization-aware training
- `inference_stage1.py`: stage-1 haze generation
- `inference_stage2.py`: standard dehazing inference
- `inference_accsamp.py`: accelerated dehazing with AccSamp
- `inference_accsamp_qat.py`: accelerated inference for QAT/plain checkpoints
- `inference_stage2_profile.py`: profiling for standard dehazing
- `inference_accsamp_profile.py`: profiling for accelerated dehazing

Evaluation:

- `eval/evaluator.py`: safety-proxy evaluator with visibility, hallucination, and photometric checks

## Method Overview

The repository works with the `DiffDehaze` pipeline, which uses a latent diffusion model with `ControlLDM + IRControlNet`.

The acceleration study in the report is centered on:

- reducing the denoising step count
- using AccSamp restart parameters `tau` and `omega`
- evaluating reduced-precision execution through QAT
- measuring the latency-quality trade-off with SPIs instead of restoration metrics alone

The report describes the denoiser as the dominant bottleneck, accounting for roughly `96%` of end-to-end latency.

## Safety-Proxy Evaluation

The report introduces a deployment-oriented evaluation framework for autonomous systems. In code, this is implemented in `eval/evaluator.py`.

The evaluator combines three groups of signals:

- `SPI-1`: visibility improvement
- `SPI-2`: hallucination risk through structure or high-frequency inflation
- `SPI-3`: photometric stability

The evaluator reports:

- overall score
- verdict: `good`, `acceptable`, `borderline`, or `high-risk`
- haze index, contrast, edge density, Laplacian variance, clipping rate, exposure, and color cast
- optional pairwise ratios when a hazy reference is available

## Installation

Clone the repository and create an environment:

```bash
git clone https://github.com/JWLMay/Project-285.git
cd Project-285

conda create -n diffdehaze-av python=3.10
conda activate diffdehaze-av
pip install -r requirements.txt
```

`requirements.txt` targets:

- Python `3.10`
- PyTorch `2.2.2`
- CUDA `11.8`
- `accelerate`
- `xformers`
- TensorBoard

## Checkpoints

Place checkpoints in `weights/` unless you change paths in the YAML configs.

Expected files include:

- `weights/v2-1_512-ema-pruned.ckpt`
- stage-1 checkpoint for haze generation
- stage-2 checkpoint for dehazing
- optional QAT checkpoint for `inference_accsamp_qat.py`

Most configs already assume the Stable Diffusion backbone checkpoint and local experiment folders.

## Data

### Stage 1

`configs/train/stage1.yaml` uses `HybridTrainingData` and expects:

- synthetic clean images in `rgb_500/`
- corresponding depth maps in `depth_500/`
- real hazy `.jpg` images for unpaired haze supervision

This matches the original HazeGen setup inherited from the base project.

### Stage 2

`configs/train/stage2.yaml` and `configs/train/stage2_qat.yaml` use `StaticPairedData` and expect:

- `hazy_folder`
- `clean_folder`

The current dataset loader assumes:

- hazy images are named like `name_index.*`
- the matching clean image is `name.jpg`

If your filenames differ, update `datasets.py`.

## Inference

Put images in `inputs/` unless you change `image_folder` in the config.

### Standard Stage-2 Inference

```bash
python inference_stage2.py --config configs/inference/stage2.yaml
```

Optional controls:

```bash
python inference_stage2.py \
  --config configs/inference/stage2.yaml \
  --steps 50 \
  --eta 1.0
```

### Accelerated AccSamp Inference

```bash
python inference_accsamp.py --config configs/inference/stage2_accsamp.yaml
```

Important config fields:

- `tau`
- `omega`
- `controlnet_path`
- `result_folder`

The default accelerated script uses `30` diffusion steps.

### QAT / Reduced-Precision Inference

```bash
python inference_accsamp_qat.py --config configs/inference/stage2_accsamp_qat.yaml
```

This script can load either:

- a plain controlnet checkpoint
- a QAT checkpoint with fake-quant weights

## Training

Configure Hugging Face Accelerate first:

```bash
accelerate config
```

### Train Stage 1

```bash
accelerate launch train_stage1.py --config configs/train/stage1.yaml
```

### Train Stage 2

```bash
accelerate launch train_stage2.py --config configs/train/stage2.yaml
```

### Train Stage 2 with QAT

```bash
accelerate launch --num_processes 1 train_stage2_QAT.py --config configs/train/stage2_qat.yaml
```

The QAT config includes:

- `qat: true`
- `qat_observer_freeze_step: 40000`

## Profiling and Evaluation

Profile inference:

```bash
python inference_stage2_profile.py --config configs/inference/stage2.yaml --steps 50
python inference_accsamp_profile.py --config configs/inference/stage2_accsamp.yaml --steps 30
```

Run the safety-proxy evaluator by importing `AVSafetyProxyEvaluator` from `eval/evaluator.py` or adapting the example block in that file.

## Experimental Setup From The Report

The final report evaluates the system with:

- model: `DiffDehaze (ControlLDM + IRControlNet)`
- train dataset: `RESIDE OTS`
- validation dataset: `nuScenes with synthesized haze`
- GPU: `NVIDIA RTX A6000`
- prompt: `"remove dense fog"`
- inference steps: `{50, 40, 30, 20, 10}`
- restart parameters: `tau in {0.8, 0.6}`, `omega in {0.6, 0.4}`
- guidance: weighted SSIM, scale `0.1`

## Important Notes

- Several scripts hard-code `CUDA_VISIBLE_DEVICES` near the top. Adjust them for your machine.
- Some paths in configs point to local experiment folders. Review YAML files before running.
- The repository still contains stage-1 haze generation code from the original project, but the final report’s main contribution is the accelerated stage-2 dehazing pipeline and its safety-aware evaluation.
- Quantization results in the report are presented mainly as an accuracy-feasibility study; hardware-level INT8 latency gains were not fully validated on specialized accelerators.

## Acknowledgment

This project is built on:

- [DiffBIR](https://github.com/XPixelGroup/DiffBIR)
- the `Learning Hazing to Dehazing` / `DiffDehaze` work by Wang et al.

## Reference

If you need the original base paper behind the inherited dehazing pipeline:

```bibtex
@inproceedings{wang2025learning,
  title={Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing},
  author={Wang, Ruiyi and Zheng, Yushuo and Zhang, Zicheng and Li, Chunyi and Liu, Shuaicheng and Zhai, Guangtao and Liu, Xiaohong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23091--23100},
  year={2025}
}
```

## License

Apache License 2.0. See [LICENSE](/Users/jwl/Downloads/Project-285/LICENSE).
