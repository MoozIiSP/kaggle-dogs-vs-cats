import hydra
from omegaconf import DictConfig, OmegaConf
from .config import EXP_CONFIG_TEMPLATE


CFG = DictConfig(EXP_CONFIG_TEMPLATE)


def test_dataset():

    dm = hydra.utils.instantiate(
        CFG.datamodule.datasets,
        transforms=CFG.datamodule.transforms,
        _recursive_=False
    )

    # FIXME: Test Numpy random bug
    dm.setup('fit')
    for _ in range(50):
        print(_)
        trainloader = iter(dm.train_dataloader())
        for _, (x, y) in enumerate(trainloader):
            # print(_, x.shape, y.shape)
            # if _ == 10:
            continue

        validloader = iter(dm.val_dataloader())
        for _, (x, y) in enumerate(validloader):
            # print(x.shape, y.shape)
            continue

    dm.setup('test')
    testloader = iter(dm.test_dataloader())
    for _, y in enumerate(testloader):
        # print(y)
        continue
