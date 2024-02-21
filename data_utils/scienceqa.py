import os
import random
import json

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from .common_utils import set_seed

OPTIONS = ['A', 'B', 'C', 'D']
MAPPING = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


def process_scienceqa(dataset_path):
    processed_file_path = os.path.join(dataset_path, 'scienceqa.json')
    set_seed(42)
    if not os.path.isfile(processed_file_path):
        print('Start processing ScienceQA dataset...')
        images_dir = os.path.join(dataset_path, 'images')
        os.makedirs(images_dir, exist_ok=True)

        sub_dataset_list = []
        for split in ['validation', 'test']:
            sub_dataset = load_dataset('derek-thomas/ScienceQA', split=split)
            sub_dataset_list.append(sub_dataset)

        dataset = concatenate_datasets(sub_dataset_list)
        questions = [row for row in dataset if row['task'] == 'closed choice' and row['image'] is not None]

        set_seed(42)

        for i, row in tqdm(enumerate(questions)):
            choices = row['choices'].copy()
            num_choices = len(choices)
            if num_choices <= 4:
                num_to_add = 4 - num_choices
                random_sampled_rows = random.choices(questions, k=num_to_add)
                for random_row in random_sampled_rows:
                    random_choice = random.choice(random_row['choices'])
                    choices.append(random_choice)
                row['answer'] = MAPPING[row['answer']]
                row.update([(opt, ch) for opt, ch in zip(OPTIONS, choices)])
            else:
                new_choices = []
                correct_answer = choices[row['answer']]
                new_choices.append(correct_answer)
                choices.remove(correct_answer)
                new_choices += random.choices(choices, k=3)
                row['answer'] = 'A'
                row.update([(opt, ch) for opt, ch in zip(OPTIONS, new_choices)])
            row['id'] = str(i)
            image_name = os.path.join(
                images_dir,
                row['id'] + '.png'
            )
            row['image_path'] = image_name
            row['image'].save(image_name)
            del row['image']
        with open(processed_file_path, 'w') as f:
            json.dump(questions, f)
    return processed_file_path
