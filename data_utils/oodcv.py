import os
import json
import shutil
import random
from urllib.request import urlretrieve

from tqdm import tqdm
import gdown

from .common_utils import set_seed


FULL_OPTIONS = list(range(6))
OPTIONS = ['A', 'B', 'C', 'D']
MAPPING = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
COUNT_MAPPING = {
    0: 'Zero',
    1: 'One',
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
}


def download_oodcv(dataset_path):
    json_url = 'https://huggingface.co/datasets/PahaII/vllm_safety_evaluation/resolve/main/safety_evaluation_benchmark_datasets.zip'
    images_url = 'https://drive.google.com/uc?id=1jq43Q0cenISIq7acW0LS-Lqghgy8exsj'

    downloaded_json_zip_file_path = os.path.join(dataset_path, 'safety_evaluation_benchmark_datasets.zip')
    downloaded_images_zip_file_path = os.path.join(dataset_path, 'images.zip')
    downloaded_json_file_path = os.path.join(
        dataset_path,
        'safety_evaluation_benchmark_datasets/ood/oodcv-vqa/oodcv-vqa.json'
    )
    if not os.path.isfile(downloaded_json_zip_file_path):
        print('Start downloading oodcv dataset...')
        urlretrieve(json_url, downloaded_json_zip_file_path)
        gdown.download(images_url, downloaded_images_zip_file_path, quiet=False)
        shutil.unpack_archive(downloaded_json_zip_file_path, dataset_path)
        shutil.unpack_archive(downloaded_images_zip_file_path, dataset_path)
    return downloaded_json_file_path


def process_oodcv(downloaded_file_path, dataset_path):
    processed_file_path = os.path.join(
        dataset_path,
        'oodcv.json'
    )
    set_seed(42)
    if not os.path.isfile(processed_file_path):
        print('Start processing oodcv dataset...')

        with open(downloaded_file_path, 'r') as f:
            questions = json.load(f)
        questions = [row for row in questions if row['answer_type'] == 'number']

        for i, row in tqdm(enumerate(questions)):
            answer = int(row['text_answer'])
            all_answers = [int(option.split(' ')[-1]) for option in row['options']]
            list_to_sample = [el for el in FULL_OPTIONS if el not in all_answers]
            sampled_elements = random.choices(list_to_sample, k=2)
            all_answers = all_answers + sampled_elements
            random.shuffle(all_answers)
            correct_answer = all_answers.index(answer)
            row['answer'] = MAPPING[correct_answer]
            row['id'] = i
            row['hint'] = None
            row.update([(opt, COUNT_MAPPING[ch]) for opt, ch in zip(OPTIONS, all_answers)])
            del row['options']
            del row['text_answer']
            image_name = os.path.join(
                dataset_path,
                row['image']
            )
            row['image_path'] = image_name

        with open(processed_file_path, 'w') as f:
            json.dump(questions, f)
    return processed_file_path