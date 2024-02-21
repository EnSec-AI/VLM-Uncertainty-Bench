from data_utils.utils import  get_dataset

DATASETS = [
    'oodcv',
    'mmbench',
    #'mmmu',
    'scienceqa',
    'seedbench'
]

def test_dataset_loading():
    for dataset_path in DATASETS:
        print(f'Starting test for dataset: {dataset_path}')
        get_dataset(dataset_path, 'datasets')

if __name__ == '__main__':
    test_dataset_loading()