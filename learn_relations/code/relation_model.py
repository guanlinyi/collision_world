from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class RelationContactStudent(nn.Module):
    """Keep spatial actor evidence before predicting a pairwise physical relation."""

    def __init__(self, width: int, grid_size: int, hidden: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU())
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.slot_mask = nn.Linear(hidden, 2)
        self.dense_physics = nn.Linear(hidden, 3)
        relation_width = hidden * 6 + 8
        self.regressor = nn.Sequential(
            nn.LayerNorm(relation_width), nn.Linear(relation_width, 128), nn.GELU(),
            nn.Dropout(.1), nn.Linear(128, 2),
        )

        axis = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("patch_xy", torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = tokens.shape[0]
        local = self.project(tokens).transpose(1, 2).reshape(
            batch, -1, self.grid_size, self.grid_size
        )
        local = self.spatial(local).flatten(2).transpose(1, 2)
        mask_logits = self.slot_mask(local).transpose(1, 2)
        dense = self.dense_physics(local)
        weights = torch.sigmoid(mask_logits)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-4)
        slots = torch.einsum("bsp,bph->bsh", weights, local)
        slot_xy = torch.einsum("bsp,pc->bsc", weights, self.patch_xy)
        predicted_presence = torch.sigmoid(dense[:, :, 0])
        predicted_depth = F.softplus(dense[:, :, 1])
        predicted_speed = F.softplus(dense[:, :, 2])
        slot_depth = torch.einsum("bsp,bp->bs", weights, predicted_depth).unsqueeze(-1)
        slot_speed = torch.einsum("bsp,bp->bs", weights, predicted_speed).unsqueeze(-1)
        slot_stats = torch.cat((slot_xy, slot_depth, slot_speed), dim=2).flatten(1)
        first, second = slots[:, 0], slots[:, 1]
        scene_mean = local.mean(dim=1)
        scene_max = local.amax(dim=1)
        relation = torch.cat((
            first, second, torch.abs(first - second), first * second,
            scene_mean, scene_max, slot_stats,
        ), dim=1)
        return {
            "contact": self.regressor(relation),
            "slot_mask_logits": mask_logits,
            "dense": dense,
            "predicted_presence": predicted_presence,
            "relation_embedding": relation,
        }


