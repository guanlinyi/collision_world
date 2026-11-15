"""Explicit SAM2.1 loader, preserving the original Hydra model construction."""
from pathlib import Path
from types import SimpleNamespace
import sys


def backend(sam_root,checkpoint,device):
    sam_root=Path(sam_root).resolve();checkpoint=Path(checkpoint).resolve()
    if not checkpoint.is_file() or not (sam_root/'sam2/configs').is_dir():raise FileNotFoundError('Explicit SAM repository/configs and checkpoint are required')
    sys.path.insert(0,str(sam_root))
    import torch,sam2
    from hydra import compose,initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.build_sam import _load_checkpoint
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not Path(sam2.__file__).resolve().is_relative_to(sam_root):raise ValueError('Imported sam2 is outside --sam-root')
    def load_predictor():
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(Path(sam2.__file__).parent/'configs'),version_base='1.2')
        config=compose(config_name='sam2.1/sam2.1_hiera_l.yaml');OmegaConf.resolve(config)
        model=instantiate(config.model,_recursive_=True);_load_checkpoint(model,str(checkpoint))
        return SAM2ImagePredictor(model.to(device).eval())
    return SimpleNamespace(torch=torch,load_predictor=load_predictor,SAM2_CHECKPOINT=checkpoint,DEVICE=device)
