EXP_CONFIG_TEMPLATE = {
    "trainer": {
        "_target_": "pytorch_lightning.Trainer",
        "gpus": 0,
        "min_epochs": 1,
        "max_epochs": 10,
        "gradient_clip_val": 0.5,
        "accumulate_grad_batches": 2,
        "weights_summary": None,
    },
    "model": {
        "net": {
            "_target_": "project.models.classifier_model.ClassifierLitModel",
            "name": "resnet18",
            "extra_cfg": None,
        },
        "transfer": {
            "pretrained": True,
            "bn_trainable": True,
            "depth": 9,
            "input_shape": (3, 224, 224),
            "head": {
                "layers": [
                    {
                        "_target_": "torch.nn.Linear",
                        "in_features": -1,
                        "out_features": 1000,
                    }
                ],
                "act": {"_target_": "torch.nn.Sigmoid"},
            },
        },
        "optimizer": {
            "lr": 0.001,
            "groups": [
                {
                    "optim": {
                        "_target_": "torch.optim.SGD",
                        "lr": 0.001,
                        "momentum": 0.9,
                        "weight_decay": 0.0005,
                    },
                    "sched": {
                        "_target_": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                        "T_0": 1,
                        "T_mult": 1,
                        "eta_min": 0,
                        "last_epoch": -1,
                    },
                }
            ],
        },
        "loss": [{"BCELoss": {"_target_": "torch.nn.BCELoss"}}],
    },
    "datamodule": {
        "datasets": {
            "_target_": "project.datamodules.kaggle_dogs_vs_cats_datamodule.DogCatDataModule",
            "data_dir": "./datasets/dogs-vs-cats-redux-kernels-edition",
    "n_channels": 3,
    "crop_size": 224,
    "num_classes": 2,
            "num_classes": 1000,
            "categories": {"dog": 1, "cat": 0},
            "batch_size": 64,
            "num_workers": 2,
            "pin_memory": True,
        },
        "transforms": {
            "type": "torchvision",
            "train": [
                {
                    "_target_": "torchvision.transforms.RandomResizedCrop",
                    "size": [224, 224],
                    "scale": [0.08, 1.0],
                    "ratio": [0.75, 1.333],
                },
                {"_target_": "torchvision.transforms.RandomVerticalFlip", "p": 0.5},
                {"_target_": "torchvision.transforms.RandomHorizontalFlip", "p": 0.5},
                {"_target_": "torchvision.transforms.ToTensor"},
            ],
            "eval": [
                {"_target_": "torchvision.transforms.Resize", "size": [224, 224]},
                {"_target_": "torchvision.transforms.ToTensor"},
            ],
        },
    },
    "callbacks": {
        "model_checkpoint": {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": "val/acc",
            "save_top_k": 1,
            "save_last": True,
            "mode": "max",
            "dirpath": "checkpoints/",
            "filename": "sample-mnist-{epoch:02d}",
        },
        "early_stopping": {
            "_target_": "pytorch_lightning.callbacks.EarlyStopping",
            "monitor": "val/acc",
            "patience": 10,
            "mode": "max",
        },
    },
    "logger": {
        "wandb": {"tags": ["best_model", "uwu"], "notes": "Description of this model."}
    },
    "seed": 42,
    "work_dir": "${hydra:runtime.cwd}",
    "data_dir": "./datasets",
    "debug": True,
    "print_config": True,
    "disable_warnings": True,
}
