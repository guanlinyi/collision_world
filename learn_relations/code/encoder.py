from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import sys
import torch

def build_encoder(checkpoint, input_frames, vjepa_root):
    if input_frames < 2 or input_frames % 2:
        raise ValueError("V-JEPA requires a positive even clip length")
    sys.path.insert(0, str(Path(vjepa_root).resolve()))
    from app.vjepa_2_1.models import vision_transformer
    encoder = vision_transformer.vit_base(
        patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2,
        use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=True,
        use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {}
    for key, value in data["ema_encoder"].items():
        state[key.replace("module.", "").replace("backbone.", "")] = value
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    allowed_unexpected = {"img_mod_embed", "video_mod_embed"}
    unexpected_real = [key for key in unexpected if key not in allowed_unexpected]
    if missing or unexpected_real:
        raise RuntimeError(f"V-JEPA权重结构不匹配：missing={missing}, unexpected={unexpected_real}")
    # 官方评测使用BF16 autocast，而不是把RoPE相关参数永久转成BF16。
    # 永久转换会使Q/K被RoPE提升为FP32、V仍为BF16，从而触发SDPA类型错误。
    return encoder.eval().cuda()


