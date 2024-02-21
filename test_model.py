import argparse

import torch

from models_utils import get_model
from data_utils import get_dataset
from data_utils.common_utils import open_json, ALL_OPTIONS
from input_utils import get_inputs


def main(args):

    tokenizer, model, image_processor, context_len = get_model(args.model)

    dataset_name = 'seedbench'
    dataset_json_path = get_dataset(dataset_name, args.data_path)
    data = open_json(dataset_json_path)
    row = data[50]

    inputs = get_inputs(row, tokenizer, image_processor, args.model, model, args.conv_mode, dataset_name)

    model_name_last_part = args.model.split('/')[-1]
    option_ids = [tokenizer.encode(opt)[-1] for opt in ALL_OPTIONS]
    if model_name_last_part.startswith(('Qwen', 'Monkey', 'MoE-LLaVA', 'Yi-VL', 'llava-v1.6-34b')) and model_name_last_part != 'Monkey-Chat':
        option_ids = [tokenizer(' ' + opt).input_ids[-1] for opt in ALL_OPTIONS]

    print("Model name: ", args.model)
    print("Options_ids: ", option_ids)

    with torch.no_grad():
        output = model(
            **inputs,
            return_dict=True,
        )
    #print(tokenizer.decode(output[0]))

    logits = output.logits.detach()
    print("Logits shape: ", logits.shape)
    logits = logits[:, -1, :]
    #logits_options = logits.squeeze(0)[option_ids].float().cpu().numpy()
    #print(logits_options)

    value, index = torch.max(logits, 1)
    print("Max index: ", index)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    args = parser.parse_args()
    main(args)
    #CUDA_VISIBLE_DEVICES=2 python -m test_models --model_name 'liuhaotian/llava-v1.5-7b' --data_path 'datasets' --conv-mode 'vicuna_v1'
