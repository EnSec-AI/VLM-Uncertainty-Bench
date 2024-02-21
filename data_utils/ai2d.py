import os
import json
import glob
import shutil
from urllib.request import urlretrieve

import tqdm

OPTIONS = ['A', 'B', 'C', 'D']


def download_ai2d(data_path, dataset_path):
    url = 'https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip'

    downloaded_zip_file_path = os.path.join(dataset_path, 'ai2d-all.zip')
    if not os.path.isfile(downloaded_zip_file_path):
        print('Start downloading AI2D dataset...')
        urlretrieve(url, downloaded_zip_file_path)
        shutil.unpack_archive(downloaded_zip_file_path, 'datasets')
    return dataset_path

def process_ai2d(dataset_path):

    question_files = glob.glob(os.path.join(dataset_path, 'questions/*.json'))
    image_dir = os.path.join(dataset_path, 'images')
    processed_file_path = os.path.join(dataset_path, 'ai2d.json')

    if not os.path.isfile(processed_file_path):
        print('Start processing AI2D dataset...')

        questions = []
        for question_file in tqdm.tqdm(question_files):
            with open(question_file, 'r') as f:
                data = json.load(f)
            image_path = data['imageName']
            for key, value in data['questions'].items():
                question = {
                    'question': key
                }
                question.update(value)
                question['id'] = value['questionId']
                choices = value['answerTexts']
                for option, choice in zip(OPTIONS, choices):
                    question[option] = choice
                question['answer'] = OPTIONS[value['correctAnswer']]
                question['hint'] = None
                question['image_path'] = os.path.join(image_dir, image_path)
                questions.append(question)

        with open(processed_file_path, 'w') as f:
            json.dump(questions, f)
    return processed_file_path