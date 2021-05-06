# run sweeper for resnet18 to fit for kaggle-dogs-vs-cats datasets.
python hydra_run.py -m hparams_search=kaggle_fix_resnet18 experiment=exp_example_full