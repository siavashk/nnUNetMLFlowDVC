import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class ModelWrapper:
    def __init__(self, predictor: nnUNetPredictor):
        self.predictor = predictor

    def predict(self, volume, params):
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

    predictor.initialize_from_trained_model_folder(path, checkpoint_name='checkpoint_best.pth')
    
    return ModelWrapper(predictor=predictor)