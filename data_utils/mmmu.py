import os
import re
import json

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

OPTIONS = ['A', 'B', 'C', 'D']

def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches

def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': None, 'question_type': data['question_type']}
    else:
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': data['image_1'], 'img_type': data['img_type'], 'hint': None,
            'topic_difficulty': data['topic_difficulty'],'subfield': data['subfield']}


def process_mmmu(dataset_path):
    processed_file_path = os.path.join(dataset_path, 'mmmu.json')
    ##unusual type of prompting
    ##need to fix, can be 2 or 3 options

    if not os.path.isfile(processed_file_path):
        print('Start processing MMMU dataset...')
        data_path = "MMMU/MMMU"
        split = 'validation'

        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_path, subject, split=split)
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        dataset = concatenate_datasets(sub_dataset_list)
        images_dir = os.path.join(dataset_path, 'images')
        os.makedirs(images_dir, exist_ok=True)

        questions = [
            process_single_sample(row) for row in dataset if row['question_type'] == 'multiple-choice'
        ]
        questions = [question for question in questions if question['image'] is not None]
        for i, row in tqdm(enumerate(questions)):
            image_name = os.path.join(
                images_dir,
                row['id'] + '.png'
            )
            row['image_path'] = image_name
            row['image'].save(image_name)
            del row['image']
            row.update([(opt, ch) for opt, ch in zip(OPTIONS, eval(row['options']))])
        with open(processed_file_path, 'w') as f:
            json.dump(questions, f)
    return processed_file_path