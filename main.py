import os
import argparse
import pickle

from tqdm import tqdm

from models_utils import get_model, get_logits
from data_utils import DATASETS, get_dataset
from data_utils.common_utils import open_json, ALL_OPTIONS
from input_utils import get_inputs


def main(args):
    """

    :param args:
    :return:
    """

    model_name = args.model
    conv_mode = args.conv_mode
    #device = torch.device(args.device)
    tokenizer, model, image_processor, context_len = get_model(model_name)

    # Check whether datasets are downloaded and download if they are not
    data_path = args.data_path
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    model_name_last_part = model_name.split('/')[-1]
    output_dir = os.path.join(args.output_dir, model_name_last_part)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset_name in DATASETS:
        print('dataset:', dataset_name)
        dataset_json_path = get_dataset(dataset_name, data_path)
        data = open_json(dataset_json_path)
        dataset_results = []
        for row in tqdm(data):
            out = {}
            inputs = get_inputs(
                row, tokenizer, image_processor, model_name, model, conv_mode, dataset_name
            )
            option_ids = [tokenizer.encode(opt)[-1] for opt in ALL_OPTIONS]
            if model_name_last_part.startswith(('Qwen', 'Monkey', 'MoE-LLaVA', 'Yi-VL', 'llava-v1.6-34b')) and model_name_last_part != 'Monkey-Chat':
                #this is correct way for these models
                option_ids = [tokenizer(' ' + opt).input_ids[-1] for opt in ALL_OPTIONS]
            logits = get_logits(model, inputs, option_ids)
            out.update(row)
            out['logits'] = logits
            dataset_results.append(out)

        out_file = os.path.join(output_dir, dataset_name + '.pkl')
        with open(out_file, "wb") as f:
            pickle.dump(dataset_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_path', type=str, default="data")
    #parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    #parser.add_argument('--file', type=str, default="xxx.json", help="Specify which dataset to use")
    parser.add_argument('--prompt_method', type=str, default="base", help="Select from 'base', 'shared', 'task'")
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--few_shot', type=int, default=0)
    parser.add_argument('--cot', action="store_true", default=False)
    args = parser.parse_args()
    #https://github.com/InternLM/InternLM-XComposer/tree/main/evaluation#q-bench


    main(args)