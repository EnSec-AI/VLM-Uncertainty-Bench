import os
import json
import shutil
from urllib.request import urlretrieve


OPTIONS = ['A', 'B', 'C', 'D']
OLD_OPTIONS = ['choice_a', 'choice_b', 'choice_c', 'choice_d']


def download_seedbench(dataset_path):
    json_url = 'https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench.json'
    zip_url = 'https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench-image.zip'

    downloaded_json_file_path = os.path.join(dataset_path, 'SEED-Bench.json')
    downloaded_zip_file_path = os.path.join(dataset_path, 'SEED-Bench-image.zip')
    if not os.path.isfile(downloaded_json_file_path):
        print('Start downloading SEEDBench dataset...')
        urlretrieve(json_url, downloaded_json_file_path)
        urlretrieve(zip_url, downloaded_zip_file_path)
        shutil.unpack_archive(downloaded_zip_file_path, dataset_path)
    return downloaded_json_file_path

def process_seedbench(downloaded_file_path):


    processed_file_path = downloaded_file_path.replace(
        'SEED-Bench.json',
        'seedbench.json'
    )
    if not os.path.isfile(processed_file_path):
        print('Start processing SEEDBench dataset...')

        with open(downloaded_file_path, 'r') as f:
            questions = json.load(f)
        dataset_path = '/'.join(downloaded_file_path.split('/')[:-1])

        questions = questions['questions']
        questions = [question for question in questions if question['question_type_id'] < 10]

        for row in questions:
            for old_option, new_option in zip(OLD_OPTIONS, OPTIONS):
                row[new_option] = row[old_option]
                del row[old_option]
            row['id'] = row['question_id']
            del row['question_id']
            row['hint'] = None
            image_name = os.path.join(
                dataset_path,
                'SEED-Bench-image',
                row['data_id']
            )
            row['image_path'] = image_name

        with open(processed_file_path, 'w') as f:
            json.dump(questions, f)
    return processed_file_path