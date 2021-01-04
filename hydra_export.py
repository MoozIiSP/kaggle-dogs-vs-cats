import hydra
import torch
import os
from omegaconf.dictconfig import DictConfig
from torchvision import models

from src.utils.utils import load_obj, reconstruct_state_dict


@hydra.main(config_path="conf", config_name="config")
def export(cfg: DictConfig) -> None:
    model_fn = getattr(models, cfg.model.class_name)
    net = model_fn(num_classes=cfg.data.num_classes)
    assert hasattr(cfg.export, 'target')
    target_path = os.path.join(hydra.utils.get_original_cwd(), cfg.export.target)
    weights = torch.load(target_path, map_location=cfg.export.device)

    state_dict = net.state_dict()
    res = reconstruct_state_dict(weights, state_dict)
    net.load_state_dict(state_dict=state_dict)

    dummy_input = torch.randn(cfg.export.input_dim, 3, 224, 224)
    input_names = cfg.export.input_tags
    output_names = cfg.export.output_tags
    torch.onnx.export(
        net, dummy_input, f"{cfg.model.class_name}-cats-vs-dogs-{cfg.export.device}.onnx",
        verbose=True, input_names=input_names, output_names=output_names)


if __name__ == "__main__":
    export()
