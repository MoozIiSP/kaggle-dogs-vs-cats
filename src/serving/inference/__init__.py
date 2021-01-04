import sys

from omegaconf.dictconfig import DictConfig
sys.path.append('/home/mooziisp/GitRepos/Kaggle-dogs-vs-cats')

from src.utils.utils import load_obj
from typing import Any, Callable, Optional
import onnx
import onnxruntime as ort
import psutil
from PIL import Image
from io import BytesIO
import albumentations as A
import numpy as np

import torch


class _SingletonType(type):
    _instances = {}
    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(_SingletonType, cls).__call__(*args, **kwds)
        return cls._instances[cls]


class Inferencer(metaclass=_SingletonType):

    def init(self, cfg):
        self.config = cfg
        if hasattr(self.config.serving, 'onnx_model'):
            self.load_model()
        else:
            raise KeyError('no onnx_model key.')

    def reset_batch_size(self):
        pass

    def register_env_var(self):
        raise NotImplemented

    def load_model(self, onnx_model: Optional[str] = None):
        loader = load_obj(self.config.serving.loader)
        self.sess = loader(self.config.serving.onnx_model if not onnx_model else onnx_model)
        self.classnames = {v: k for k, v in self.config.data.categories.items()}

    def eval(self, data):
        im = self._data_pre_process(data)
        outputs = self.sess.run(None, {
            'input': im.unsqueeze(0).numpy()
        })
        pred = outputs[0]
        # FIXME: This is a special method, so you should make it to be a function
        #   and only eval one image once.
        print(pred)
        pred = torch.softmax(torch.as_tensor(pred%10).type(torch.float64), dim=1).detach().numpy()
        print(pred)
        # return prob map
        return {
            self.classnames[0]: pred[0][1],
            self.classnames[1]: 1 - pred[0][1],
        }

    def _data_pre_process(self, data):
        augs_list = [load_obj(i['class_name'])(**i['params']) for i in self.config['augmentation']['valid']['augs']]
        augs = A.Compose(augs_list)

        # check data whether is bytes or numpy
        if type(data) is bytes:
            data_dict = {
                'image': np.array(Image.open(BytesIO(data)), dtype=np.float32),
            }
        else:
            data_dict = {
                'image': data,
            }
        image = augs(**data_dict)['image']
        return image

def onnx_loader(model_path):
    # Load the ONNX model
    model = onnx.load_model(model_path)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)
    return model


def onnxruntime_loader(model_path):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = model_path
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    return ort.InferenceSession(model_path, sess_options)


if __name__ == "__main__":
    import hydra
    import numpy as np

    @hydra.main(config_path="../../../conf", config_name="config")
    def test(cfg: DictConfig):
        inferencer = Inferencer()
        inferencer.init(cfg)
        inferencer.load_model()
        inputs = np.random.randn(224, 224, 3).astype(np.float32)
        outputs = inferencer.eval(inputs)
        print(outputs)

    test()