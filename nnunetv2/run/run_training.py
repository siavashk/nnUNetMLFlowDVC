import os
import socket
import shutil
import sys
from typing import Union, Optional
import yaml

import mlflow
import numpy as np

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed, mlflow_tracking_token, mlflow_tracking_uri, get_identity_token
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.run import wrapper_model
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn


def load_yaml(filename: str):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)
    

def copy_essential_files_for_inference_to_temp(output_folder_base: str, fold: Union[int, str]):
    data_path = "/tmp/mlflow-model"

    if os.path.isdir(data_path):
        shutil.rmtree(data_path)

    os.mkdir(data_path)

    shutil.copy(join(output_folder_base, "dataset.json"), join(data_path, "dataset.json"))
    shutil.copy(join(output_folder_base, "dataset_fingerprint.json"), join(data_path, "dataset_fingerprint.json"))
    shutil.copy(join(output_folder_base, "plans.json"), join(data_path, "plans.json"))

    if isinstance(fold, int):
        fold_path = join(data_path, f"fold_{fold}")
        os.mkdir(fold_path)
        shutil.copy(join(output_folder_base, f"fold_{fold}", "checkpoint_best.pth"), join(fold_path, "checkpoint_best.pth"))
    elif fold == "crossvalidation":
        for i in range(5):
            fold_path = join(data_path, f"fold_{i}")
            os.mkdir(fold_path)
            shutil.copy(join(output_folder_base, f"fold_{i}", "checkpoint_best.pth"), join(fold_path, "checkpoint_best.pth"))
    else:
        raise ValueError(f"Error: unknown value for fold {fold}. Expected either crossvalidation or an int between 0 and 4 (inclusive).")
    
    return data_path


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed: bool = False,
                          device: torch.device = torch.device('cuda')):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                    'nnUNetTrainer'


    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    dvc_yaml = load_yaml(join(preprocessed_dataset_folder_base, 'dataset_info.dvc'))
    dataset_json = {**dataset_json, **dvc_yaml}

    if mlflow_tracking_uri:
        os.environ['MLFLOW_TRACKING_TOKEN'] = get_identity_token()
        mlflow.log_params(dataset_json)

    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device)
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                               f"continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, use_compressed, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p,
                                           use_compressed)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


def train_one_fold_synchronously(dataset_name_or_id, configuration, fold, trainer_class_name, 
                                plans_identifier, use_compressed_data, device, disable_checkpointing,
                                continue_training, only_run_validation, pretrained_weights, val_with_best, export_validation_probabilities):
    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                        plans_identifier, use_compressed_data, device=device)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not only_run_validation:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(export_validation_probabilities)

    return nnunet_trainer


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 use_compressed_data: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda')):
    if isinstance(fold, str):
        if fold == 'all':
            pass
        elif fold == "crossvalidation":
            pass
        else:
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     use_compressed_data,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus),
                 nprocs=num_gpus,
                 join=True)
    else:
        if isinstance(fold, int):
            nnunet_trainer = train_one_fold_synchronously(dataset_name_or_id, configuration, fold, trainer_class_name, 
                                plans_identifier, use_compressed_data, device, disable_checkpointing,
                                continue_training, only_run_validation, pretrained_weights, val_with_best, export_validation_probabilities)
        elif fold == "crossvalidation":
            for i in range(5):
                nnunet_trainer = train_one_fold_synchronously(dataset_name_or_id, configuration, i, trainer_class_name, 
                                plans_identifier, use_compressed_data, device, disable_checkpointing,
                                continue_training, only_run_validation, pretrained_weights, val_with_best, export_validation_probabilities)

        if mlflow_tracking_uri:
            os.environ["MLFLOW_TRACKING_TOKEN"] = get_identity_token()
            data_path = copy_essential_files_for_inference_to_temp(nnunet_trainer.output_folder_base, fold)
            conda_env = {
                "name": "nnunetv2.6",
                "channels": ["conda-forge"],
                "dependencies": [
                    f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    {
                        "pip": [
                            "torch==2.0.1",
                            "git+https://github.com/MIC-DKFZ/nnUNet"
                        ]
                    }
                ]
            }

            input_schema = mlflow.types.schema.Schema([
                mlflow.types.ColSpec(type="string", name="base64_array"),
                mlflow.types.ColSpec(type="integer", name="depth"),
                mlflow.types.ColSpec(type="integer", name="width"),
                mlflow.types.ColSpec(type="integer", name="height"),
                mlflow.types.ColSpec(type="double", name="spacing_z"),
                mlflow.types.ColSpec(type="double", name="spacing_y"),
                mlflow.types.ColSpec(type="double", name="spacing_x")
            ])
            
            output_schema = mlflow.types.schema.Schema([
                mlflow.types.TensorSpec(np.dtype(np.uint16), (-1, -1, -1), name="label")
            ])
            
            signature = mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)
            
            mlflow.pyfunc.log_model(
                artifact_path="model",
                loader_module="wrapper_model",
                data_path=data_path,
                conda_env=conda_env,
                signature=signature,
                code_path=[wrapper_model.__file__]
            )

            if isinstance(fold, int): 
                mlflow.log_artifact(join(nnunet_trainer.output_folder_base, f"fold_{fold}", "progress.png"))
                mlflow.log_artifact(join(nnunet_trainer.output_folder_base, f"fold_{fold}", "debug.json"))
                mlflow.log_artifacts(join(nnunet_trainer.output_folder_base, f"fold_{fold}", "validation"), artifact_path="validation")
            elif fold == "crossvalidation":
                for i in range(5):
                    mlflow.log_artifact(join(nnunet_trainer.output_folder_base, f"fold_{i}", "progress.png"), artifact_path=f"fold_{i}")
                    mlflow.log_artifact(join(nnunet_trainer.output_folder_base, f"fold_{i}", "debug.json"), artifact_path=f"fold_{i}")
                    mlflow.log_artifacts(join(nnunet_trainer.output_folder_base, f"fold_{i}", "validation"), artifact_path="validation")


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    parser.add_argument('--experiment', type=str, required=False,
                    help="Use this to set the mlflow experiment name.")
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    if args.experiment:
        if mlflow_tracking_uri and os.environ.get("MLFLOW_TRACKING_TOKEN"):
            mlflow.set_experiment(experiment_name=args.experiment)
        else:
            print("Unable to create / use existing MLflow experiment because MLFLOW_TRACKING_TOKEN and MLFLOW_TRACKING_URI are not set.")
            print("Ignoring the experiment argument.")


    with mlflow.start_run():
        mlflow.log_params(vars(args))
        run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                    args.num_gpus, args.use_compressed, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
                    device=device)


if __name__ == '__main__':
    run_training_entry()
