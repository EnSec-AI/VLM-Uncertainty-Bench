import os

from .mmbench import process_mmbench, download_mmbench
from .seedbench import process_seedbench, download_seedbench
from .scienceqa import process_scienceqa
from .mmmu import process_mmmu
from .oodcv import download_oodcv, process_oodcv
from .ai2d import download_ai2d, process_ai2d

DATASETS = [
    'mmbench',
    'oodcv',
    'scienceqa',
    'seedbench',
    'ai2d'
]

SEEDBENCH_CATS = {
    'Scene Understanding': 1,
    'Instance Identity': 2,
    'Instance Location': 3,
    'Instance Attributes': 4,
    'Instances Counting': 5,
    'Spatial Relation': 6,
    'Instance Interaction': 7,
    'Visual Reasoning': 8,
    'Text Understanding': 9,
}

OODCV_CATS = [
    'weather',
    'context',
    'occlusion',
    'iid',
    'texture',
    'shape',
    'pose'
]

def get_dataset(dataset_name, data_path):
    if dataset_name == 'mmbench':
        dataset_path = os.path.join(data_path, dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        downloaded_file_path = download_mmbench(dataset_path)
        processed_file_path = process_mmbench(downloaded_file_path)
    elif dataset_name == 'seedbench':
        dataset_path = os.path.join(data_path, dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        downloaded_file_path = download_seedbench(dataset_path)
        processed_file_path = process_seedbench(downloaded_file_path)
    elif dataset_name == 'scienceqa':
        dataset_path = os.path.join(data_path, dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        processed_file_path = process_scienceqa(dataset_path)
    elif dataset_name == 'mmmu':
        dataset_path = os.path.join(data_path, dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        processed_file_path = process_mmmu(dataset_path)
    elif dataset_name == 'oodcv':
        dataset_path = os.path.join(data_path, dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        downloaded_file_path = download_oodcv(dataset_path)
        processed_file_path = process_oodcv(downloaded_file_path, dataset_path)
    elif dataset_name == 'ai2d':
        dataset_path = os.path.join(data_path, dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        dataset_path = download_ai2d(data_path, dataset_path)
        processed_file_path = process_ai2d(dataset_path)
    else:
        raise ValueError('Unrecognized dataset')
    return processed_file_path