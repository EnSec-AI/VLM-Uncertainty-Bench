import os
import random
import json
from urllib.request import urlretrieve

from tqdm import tqdm
import pandas as pd
from llava.mm_utils import load_image_from_base64
#TODO: extract this function

from .common_utils import set_seed, get_options

OPTIONS = ['A', 'B', 'C', 'D']


def download_mmbench(dataset_path):
    downloaded_file_path = os.path.join(dataset_path, 'mmbench_dev_20230712.tsv')
    if not os.path.isfile(downloaded_file_path):
        print('Start downloading MMBench dataset...')
        url = 'https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv'
        urlretrieve(url, downloaded_file_path)
    return downloaded_file_path

def process_mmbench(downloaded_file_path):
    processed_file_path = downloaded_file_path.replace(
        'mmbench_dev_20230712.tsv',
        'mmbench.json'
    )
    set_seed(42)
    if not os.path.isfile(processed_file_path):
        print('Start processing MMBench dataset...')
        questions = []
        df = pd.read_table(downloaded_file_path)

        images_dir = os.path.join(
            '/'.join(downloaded_file_path.split('/')[:-1]),
            'images'
        )
        os.makedirs(images_dir, exist_ok=True)

        #from tsv to list, save images
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image = load_image_from_base64(row['image'])
            image_name = os.path.join(
                images_dir,
                str(row['index']) + '.png'
            )
            row['image_path'] = image_name
            image.save(image_name)
            del row['image']
            del row['comment']
            row['id'] = row['index']
            question = row.to_dict()
            questions.append(question)

        #add random options when len(options) < 4
        for row in questions:
            options = get_options(row, OPTIONS)
            num_choices = len(options)
            if num_choices <= 4:
                num_to_add = 4 - num_choices
                random_sampled_rows = random.choices(questions, k=num_to_add)
                for index, random_row in enumerate(random_sampled_rows):
                    samples = [value for key, value in random_row.items() if key in OPTIONS]
                    sample = random.choice(samples)
                    row[OPTIONS[-(index+1)]] = sample

        with open(processed_file_path, 'w') as f:
            json.dump(questions, f)
    return processed_file_path
