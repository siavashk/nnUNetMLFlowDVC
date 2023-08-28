import base64
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class ModelWrapper:
    def __init__(self, predictor: nnUNetPredictor):
        self.predictor = predictor

    def predict(self, data):
        base64_array = data.iloc[0]["base64_array"]
        depth = data.iloc[0]["depth"]
        width = data.iloc[0]["width"]
        height = data.iloc[0]["height"]
        spacing_z = data.iloc[0]["spacing_z"]
        spacing_y = data.iloc[0]["spacing_y"]
        spacing_x = data.iloc[0]["spacing_x"]

        decoded_bytes = base64.b64decode(base64_array)
        volume = np.frombuffer(decoded_bytes, dtype=np.float16).reshape((1, depth, width, height))
        
        params = {"spacing": [spacing_z, spacing_y, spacing_x]}
        
        return self.predictor.predict_single_npy_array(volume, params, None, None, False)


def _load_pyfunc(path):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(path, use_folds=None, checkpoint_name='checkpoint_best.pth')
    
    return ModelWrapper(predictor=predictor)