import glob
import os
from argparse import ArgumentParser

import cv2
import torch
import torch.nn as nn
import torchvision
from omegaconf import OmegaConf
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    disable_observer,
    enable_fake_quant,
)
from torch.ao.quantization.observer import MovingAveragePerChannelMinMaxObserver
from torchvision.transforms import InterpolationMode, Resize, ToTensor
from tqdm import tqdm

from diffbir.model import ControlLDM, Diffusion
from diffbir.pipeline import pad_to_multiples_of
from diffbir.sampler import SpacedSampler
from diffbir.utils.common import instantiate_from_config
from diffbir.utils.cond_fn import Guidance


class FakeQuantConv2d(nn.Conv2d):
    """
    Conv2d module used during manual Conv-only QAT training.
    This recreates the same parameter/buffer structure expected by the QAT checkpoint.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.weight_fake_quant = FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )()

    @classmethod
    def from_conv(cls, conv: nn.Conv2d):
        new_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
        )
        new_conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            new_conv.bias.data.copy_(conv.bias.data)
        return new_conv

    def forward(self, x):
        q_weight = self.weight_fake_quant(self.weight)
        return self._conv_forward(x, q_weight, self.bias)


def replace_controlnet_conv_only_qat(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, FakeQuantConv2d):
            setattr(module, name, FakeQuantConv2d.from_conv(child))
        else:
            replace_controlnet_conv_only_qat(child)


def unwrap_controlnet_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "controlnet" in ckpt_obj:
        return ckpt_obj["controlnet"]
    return ckpt_obj


def normalize_qat_state_dict_keys(state_dict):
    """
    Support both training variants:
      1) subclass-style keys: block.0.weight
      2) wrapper-style keys:  block.0.conv.weight
    Convert wrapper-style conv keys to subclass-style keys expected by FakeQuantConv2d.
    """
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if ".conv.weight" in new_key:
            new_key = new_key.replace(".conv.weight", ".weight")
        if ".conv.bias" in new_key:
            new_key = new_key.replace(".conv.bias", ".bias")
        normalized[new_key] = value
    return normalized


def load_qat_or_plain_controlnet(cldm: ControlLDM, ckpt_path: str, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = unwrap_controlnet_state_dict(ckpt)

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format at {ckpt_path}: {type(state_dict)}")

    has_fake_quant = any("weight_fake_quant" in k for k in state_dict.keys())

    if has_fake_quant:
        replace_controlnet_conv_only_qat(cldm.controlnet)
        state_dict = normalize_qat_state_dict_keys(state_dict)
        missing, unexpected = cldm.controlnet.load_state_dict(state_dict, strict=strict)
        cldm.controlnet.apply(disable_observer)
        cldm.controlnet.apply(enable_fake_quant)
        print(f"loaded QAT controlnet checkpoint from {ckpt_path}")
        print(f"missing keys: {missing}")
        print(f"unexpected keys: {unexpected}")
        return

    cldm.load_controlnet_from_ckpt(state_dict)
    print(f"loaded plain controlnet checkpoint from {ckpt_path}")


@torch.no_grad()
def main(args) -> None:
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.inference.get("device", "cuda:0"))
    os.makedirs(cfg.inference.result_folder, exist_ok=True)

    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)

    sd = torch.load(cfg.inference.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    print(
        f"strictly load pretrained SD weight from {cfg.inference.sd_path}\n"
        f"unused weights: {unused}\n"
        f"missing weights: {missing}"
    )

    load_qat_or_plain_controlnet(
        cldm,
        cfg.inference.controlnet_path,
        strict=cfg.inference.qat.get("strict", False),
    )

    cldm.eval().to(device)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    diffusion.eval()

    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    guidance: Guidance = instantiate_from_config(cfg.guidance)

    min_resize = cfg.inference.get("min_resize", 512)
    rescaler = Resize(min_resize, interpolation=InterpolationMode.BICUBIC, antialias=True)

    prompt = cfg.inference.get("prompt", "remove dense fog")
    steps = cfg.inference.get("steps", 30)
    cfg_scale = cfg.inference.get("cfg_scale", 1.0)

    image_names = sorted(
        [
            os.path.basename(name)
            for ext in ("*.jpg", "*.jpeg", "*.png")
            for name in glob.glob(os.path.join(cfg.inference.image_folder, ext))
        ]
    )

    if not image_names:
        raise FileNotFoundError(
            f"No input images found under: {cfg.inference.image_folder}"
        )

    for image_name in tqdm(image_names):
        image = cv2.imread(os.path.join(cfg.inference.image_folder, image_name))
        if image is None:
            print(f"warning: failed to read {image_name}, skipping")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image).unsqueeze(0)

        _, _, h, w = image.shape
        if h < min_resize or w < min_resize:
            image = rescaler(image)
        _, _, h_, w_ = image.shape

        image = pad_to_multiples_of(image, multiple=64).to(device)
        cond = cldm.prepare_condition(image, [prompt])

        z = sampler.accsamp(
            model=cldm,
            device=device,
            steps=steps,
            x_size=cond["c_img"].shape,
            cond=cond,
            uncond=None,
            cond_fn=guidance,
            hazy=image,
            diffusion=diffusion,
            cfg_scale=cfg_scale,
            progress=False,
            proportions=[cfg.inference.tau, cfg.inference.omega],
        )

        result = (cldm.vae_decode(z) + 1) / 2
        result = result[:, :, :h_, :w_].clip(0.0, 1.0)
        result = Resize(
            (h, w), interpolation=InterpolationMode.BICUBIC, antialias=True
        )(result)
        torchvision.utils.save_image(
            result.squeeze(0),
            os.path.join(
                cfg.inference.result_folder,
                f"{image_name.rsplit('.', 1)[0]}.png",
            ),
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
