import hydra

# from hydra.experimental import initialize, compose
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from torchinfo import summary
from .config import EXP_CONFIG_TEMPLATE


CFG = DictConfig(EXP_CONFIG_TEMPLATE)


def test_classifier_model():
    seed_everything(42)

    # FIXME:
    model = hydra.utils.instantiate(CFG.model.net, cfg=CFG.model, _recursive_=False)

    summary(
        model,
        input_size=[
            1,
            CFG.datamodule.datasets.n_channels,
            *CFG.datamodule.datasets.crop_size,
        ],
    )


def test_configure_optimizers():
    seed_everything(42)

    model = hydra.utils.instantiate(CFG.model.net, cfg=CFG.model, _recursive_=False)

    model.configure_optimizers()
