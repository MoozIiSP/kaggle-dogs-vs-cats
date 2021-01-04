import hydra
import torch
import os
from omegaconf.dictconfig import DictConfig
from torchvision import models

from src.utils.utils import load_obj, reconstruct_state_dict
from src.lightning_classes.lightning_transfer import LitTransferLearning
from PIL import Image
from torchvision import transforms


@hydra.main(config_path="conf", config_name="config")
def export(cfg: DictConfig) -> None:
    # model_fn = getattr(models, cfg.model.class_name)
    # net = model_fn(num_classes=cfg.data.num_classes)
    # weights = torch.load('/home/mooziisp/GitRepos/kaggle-dogs-vs-cats/outputs/2020-12-07/06-51-15.pth',
    #                      map_location=cfg.export.device)

    # state_dict = net.state_dict()
    # res = reconstruct_state_dict(weights, state_dict)
    # net.load_state_dict(state_dict=state_dict)
    model = LitTransferLearning(cfg)
    model.load_from_checkpoint("/home/mooziisp/GitRepos/kaggle-dogs-vs-cats/outputs/2020-12-07/06-51-15/default/0/checkpoints/epoch=24.ckpt")

    # Snippet code
    im = Image.open('/home/mooziisp/GitRepos/kaggle-dogs-vs-cats/datasets/dog.0.jpg')
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    x = t(im).unsqueeze(0)
    with torch.no_grad():
        preds = torch.softmax(model(x), dim=1)
    print(preds)
    


if __name__ == "__main__":
    export()
