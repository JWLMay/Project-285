import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import glob
import time
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from diffbir.sampler import SpacedSampler
from diffbir.model import ControlLDM, Diffusion
from diffbir.pipeline import pad_to_multiples_of
from diffbir.utils.common import instantiate_from_config
from torchvision.transforms import ToTensor, Resize, InterpolationMode
from torch.profiler import profile, record_function, ProfilerActivity


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────

def reset_peak_memory(device):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def peak_memory_mb(device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def current_memory_mb(device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / 1024 ** 2


class CudaTimer:
    """Accurate GPU wall-clock timer using CUDA events."""
    def __init__(self, device):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        self.start_event.record()
        return self

    def __exit__(self, *_):
        self.end_event.record()
        torch.cuda.synchronize(self.device)
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        self.elapsed_s  = self.elapsed_ms / 1000.0


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

@torch.no_grad()
def main(args) -> None:
    device = torch.device("cuda:0")
    cfg    = OmegaConf.load(args.config)
    os.makedirs(cfg.inference.result_folder, exist_ok=True)

    # ── Model loading ──────────────────────────────────────────────────────
    print("\n[Profile] Loading models …")
    t0 = time.perf_counter()

    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.inference.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    print(
        f"  Pretrained SD loaded from {cfg.inference.sd_path}\n"
        f"  unused: {unused}\n"
        f"  missing: {missing}"
    )

    cldm.load_controlnet_from_ckpt(
        torch.load(cfg.inference.controlnet_path, map_location="cpu")
    )
    print(f"  ControlNet loaded from {cfg.inference.controlnet_path}")

    cldm.eval().to(device)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )

    model_load_time = time.perf_counter() - t0
    print(f"[Profile] Model load time : {model_load_time:.2f} s")

    # Model structure
    print("\n[Profile] ── Model Structure ──────────────────────────────────")
    print(cldm)
    print("───────────────────────────────────────────────────────────────\n")

    # Parameter count
    total_params   = sum(p.numel() for p in cldm.parameters())
    trainable_params = sum(p.numel() for p in cldm.parameters() if p.requires_grad)
    print(f"[Profile] Total parameters    : {total_params  / 1e6:.2f} M")
    print(f"[Profile] Trainable parameters: {trainable_params / 1e6:.2f} M")

    # Memory after model load
    mem_after_load = current_memory_mb(device)
    print(f"[Profile] GPU memory after model load: {mem_after_load:.1f} MB\n")

    # ── Image list ────────────────────────────────────────────────────────
    rescaler = Resize(512, interpolation=InterpolationMode.BICUBIC, antialias=True)

    image_names = [
        os.path.basename(name)
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for name in glob.glob(os.path.join(cfg.inference.image_folder, ext))
    ]

    if len(image_names) == 0:
        print("[Profile] No images found – exiting.")
        return

    # Limit the number of images for profiling (override via --max_images)
    if args.max_images > 0:
        image_names = image_names[: args.max_images]
    print(f"[Profile] Profiling on {len(image_names)} image(s).\n")

    # ── Warm-up (not timed) ───────────────────────────────────────────────
    if args.warmup > 0:
        print(f"[Profile] Warm-up: {args.warmup} step(s) …")
        dummy_img = torch.rand(1, 3, 512, 512, device=device)
        for _ in range(args.warmup):
            cond = cldm.prepare_condition(dummy_img, ['remove dense fog'])
            z = sampler.sample(
                model=cldm, device=device, steps=args.steps,
                x_size=cond['c_img'].shape, cond=cond,
                uncond=None, cfg_scale=1., progress=False, eta=args.eta,
            )
        torch.cuda.synchronize(device)
        print("[Profile] Warm-up done.\n")

    # ── Per-image timing accumulators ─────────────────────────────────────
    times_preprocess    = []
    times_cond          = []
    times_sample        = []
    times_vae_decode    = []
    times_total         = []
    peak_mems           = []

    # ── (Optional) torch.profiler trace ──────────────────────────────────
    profiler_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(cfg.inference.result_folder, "profiler_trace")
            ),
            schedule=torch.profiler.schedule(
                wait=0, warmup=0, active=args.trace_images, repeat=1
            ),
        )
        if args.trace_images > 0
        else None
    )

    if profiler_ctx is not None:
        profiler_ctx.__enter__()

    # ── Main inference loop ───────────────────────────────────────────────
    for idx, image_name in enumerate(tqdm(image_names, desc="Profiling")):

        reset_peak_memory(device)

        # --- Pre-processing ---
        with CudaTimer(device) as t_pre:
            with record_function("preprocess"):
                image = cv2.imread(os.path.join(cfg.inference.image_folder, image_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = ToTensor()(image).unsqueeze(0)
                _, _, h, w = image.shape
                if h < 512 or w < 512:
                    image = rescaler(image)
                _, _, h_, w_ = image.shape
                image = pad_to_multiples_of(image, multiple=64).to(device)

        # --- Condition preparation ---
        with CudaTimer(device) as t_cond:
            with record_function("prepare_condition"):
                cond = cldm.prepare_condition(image, ['remove dense fog'])

        # --- Diffusion sampling ---
        with CudaTimer(device) as t_samp:
            with record_function("diffusion_sample"):
                z = sampler.sample(
                    model=cldm, device=device, steps=args.steps,
                    x_size=cond['c_img'].shape, cond=cond,
                    uncond=None, cfg_scale=1., progress=False, eta=args.eta,
                )

        # --- VAE decode + save ---
        with CudaTimer(device) as t_vae:
            with record_function("vae_decode"):
                result = (cldm.vae_decode(z) + 1) / 2
                result = result[:, :, :h_, :w_].clip(0., 1.)
                result = Resize(
                    (h, w), interpolation=InterpolationMode.BICUBIC, antialias=True
                )(result)

        torchvision.utils.save_image(
            result.squeeze(0),
            os.path.join(cfg.inference.result_folder, f'{image_name[:-4]}.png'),
        )

        # Accumulate
        total_ms = t_pre.elapsed_ms + t_cond.elapsed_ms + t_samp.elapsed_ms + t_vae.elapsed_ms
        times_preprocess.append(t_pre.elapsed_ms)
        times_cond.append(t_cond.elapsed_ms)
        times_sample.append(t_samp.elapsed_ms)
        times_vae_decode.append(t_vae.elapsed_ms)
        times_total.append(total_ms)
        peak_mems.append(peak_memory_mb(device))

        if profiler_ctx is not None:
            profiler_ctx.step()

    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)
        print(f"\n[Profile] TensorBoard trace saved to "
              f"{os.path.join(cfg.inference.result_folder, 'profiler_trace')}")

    # ── Summary ───────────────────────────────────────────────────────────
    n = len(times_total)
    print("\n" + "=" * 60)
    print(f"  PROFILING SUMMARY  ({n} image(s), {args.steps} diffusion steps)")
    print("=" * 60)

    def stats(arr, label, unit="ms"):
        a = np.array(arr)
        print(f"  {label:<28}  avg={a.mean():8.1f} {unit}"
              f"  std={a.std():6.1f}  min={a.min():8.1f}  max={a.max():8.1f}")

    stats(times_preprocess,  "Pre-processing")
    stats(times_cond,        "Condition preparation")
    stats(times_sample,      "Diffusion sampling")
    stats(times_vae_decode,  "VAE decode")
    stats(times_total,       "Total per image")
    print()
    stats(peak_mems,         "Peak GPU memory",  unit="MB")
    print()

    avg_total_s = np.mean(times_total) / 1000.0
    print(f"  Throughput            :  {1.0 / avg_total_s:.3f} images/s")
    print(f"  Model load time       :  {model_load_time:.2f} s")
    print(f"  GPU memory (model)    :  {mem_after_load:.1f} MB")
    print("=" * 60)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config",      type=str,  required=True,
                        help="Path to inference YAML config.")
    parser.add_argument("--steps",       type=int,  default=50,
                        help="Number of diffusion sampling steps (default: 50).")
    parser.add_argument("--warmup",      type=int,  default=1,
                        help="Number of warm-up passes before timing (default: 1).")
    parser.add_argument("--max_images",  type=int,  default=0,
                        help="Max images to profile; 0 = all (default: 0).")
    parser.add_argument("--trace_images", type=int, default=0,
                        help="Save a torch.profiler trace for the first N images; "
                             "0 = disabled (default: 0). "
                             "Traces can be viewed with TensorBoard.")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="DDIM eta: 0.0=deterministic DDIM, 1.0=stochastic DDPM (default: 1.0).")
    args = parser.parse_args()
    main(args)
