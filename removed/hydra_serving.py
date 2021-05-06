import os
import hydra
from omegaconf.dictconfig import DictConfig
from project.serving import app, gInferencer


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: DictConfig) -> None:
    # redirect to root
    os.chdir('../../../')
    gInferencer.init(cfg)
    app.run(host=cfg.serving.host, port=cfg.serving.port, debug=cfg.serving.debug)


if __name__ == '__main__':
    run_model()
    