import os
import json
import argparse

import pandas as pd
import numpy as np

from data_utils import DATASETS, SEEDBENCH_CATS, OODCV_CATS


def make_table_by_key(result_data, key):
    all_columns = ['model_name'] + DATASETS + ['avg']
    df = pd.DataFrame(columns=all_columns)
    for model_key in result_data:
        result_list = [model_key] + result_data[model_key][key] + [np.mean(result_data[model_key][key])]
        df.loc[len(df.index)] = result_list
    return df


def make_table_by_key_seedbench(result_data, key):
    cats_indices = [1,2,3,4, 5]
    cats = {key:value for key, value in SEEDBENCH_CATS.items() if value in cats_indices}
    all_columns = ['model_name'] + list(cats.keys()) + ['avg']
    df = pd.DataFrame(columns=all_columns)
    for model_key in result_data:
        cats_result = []
        for index, res in enumerate(result_data[model_key][key]):
            if index+1 in cats_indices:
                cats_result.append(res)
        result_list = [model_key] + cats_result + [np.mean(cats_result)]
        df.loc[len(df.index)] = result_list
    return df


def make_table_by_key_oodcv(result_data, key):
    all_columns = ['model_name'] + OODCV_CATS + ['avg']
    df = pd.DataFrame(columns=all_columns)
    for model_key in result_data:
        cats_result = []
        for index, res in enumerate(result_data[model_key][key]):
            cats_result.append(res)
        result_list = [model_key] + cats_result + [np.mean(cats_result)]
        df.loc[len(df.index)] = result_list
    return df


def main(args):
    os.makedirs(args.dir_to_write, exist_ok=True)
    with open(args.result_path, 'r') as f:
        result_data = json.load(f)
    for key in [
        'acc', 'coverage', 'set_sizes', 'uacc',
        'E_ratio', 'F_ratio',
        'coverage_all_LAC', 'set_sizes_LAC', 'uacc_LAC',
        'coverage_all_APS', 'set_sizes_APS', 'uacc_APS',
        'ece', 'mce'
    ]:
        if args.mode == "all":
            df = make_table_by_key(result_data, key)
        elif args.mode == "seedbench":
            df = make_table_by_key_seedbench(result_data, key)
        elif args.mode == "oodcv":
            df = make_table_by_key_oodcv(result_data, key)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        df.to_csv(f'{args.dir_to_write}/{key}.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--dir_to_write", type=str, required=True)
    parser.add_argument("--mode", choices=["all", "seedbench", "oodcv"], default="all")
    args = parser.parse_args()

    main(args)
    #python -m make_tables --result_path full_result02.json --dir_to_write 'tables02'
    #python -m make_tables --result_path full_result_seedbench.json --mode 'seedbench' --dir_to_write 'seedbench_tables'
    #python -m make_tables --result_path full_result_oodcv.json --mode 'oodcv' --dir_to_write 'oodcv_tables'

