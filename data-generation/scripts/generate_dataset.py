import torch
import yaml
import random
import numpy as np
from yaml import Loader
import warnings
import copy
from config import *
from task import *
import argparse
import os

def get_idx_select_0(num_exper, frac_0):
    if (frac_0 > 0.05):
        warnings.warn("frac_0 is greater than 0.05")
    idx_lst_0 = [i for i in range(num_exper)]
    idx_select_0 = np.random.choice(idx_lst_0, int(num_exper * frac_0), replace=False).tolist()
    return idx_select_0

def get_idx_select_top5(num_exper, frac_top5, idx_select_0=None):
    if (frac_top5 > 0.05):
        warnings.warn("frac_top5 is greater than 0.05")
    if idx_select_0 is None:
        idx_lst_top5 = [i for i in range(num_exper)]
    else:
        idx_lst_top5 = [i for i in range(num_exper) if i not in idx_select_0]
    idx_select_top5 = np.random.choice(idx_lst_top5, int(num_exper * frac_top5), replace=False).tolist()
    return idx_select_top5

def main(yaml_path, seed, verbose, only_check_task):
    if only_check_task and not verbose:
        raise ValueError("verbose should be True if only_check_task is True")
    with open(yaml_path, 'r') as fin:
        args = yaml.load(fin, Loader)

    if args['cls'] == 'SimpleDictRetrieval':
        dataconfig_cls = SimpleDictRetrievalDataConfig
        taskconfig_cls = SimpleDictRetrievalTaskConfig
        task_cls = DictRetrieval
        exper_cls = DictRetrievalExper
    else:
        raise NotImplementedError
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["tmp_seed"] = str(seed)

    dataconfig = dataconfig_cls(**args['data_kwargs'])
    taskconfig = taskconfig_cls(**args['task_kwargs'])

    task = task_cls(dataconfig, taskconfig, args['tokenizer'])
    p = task.get_prompt(0)
    if (verbose):
        print(p+"\n")
        print(f"num tokens:{task.get_prompt_len(p)}")
        if (only_check_task):
            exit()

    if 'alias' in args.keys():
        filename = args['alias']
    else:
        filename = exper_train.get_str()

    exper_train = exper_cls(args['num_train'], args['tokenizer'], dataconfig, taskconfig)
    exper_eval = exper_cls(args['num_eval'], args['tokenizer'], dataconfig, taskconfig)

    if args['frac_0'] > 0 or args['frac_top5'] > 0:
        train_idx_select_0 = None
        eval_idx_select_0 = None
        new_train_gold_idx_lst = copy.deepcopy(exper_train.gold_idx_lst)
        new_eval_gold_idx_lst = copy.deepcopy(exper_eval.gold_idx_lst)
        if (verbose):
            print(f"train_gold_idx_lst: {exper_train.gold_idx_lst}\neval_gold_idx_lst: {exper_eval.gold_idx_lst}")
    if args['frac_0'] > 0:
        train_idx_select_0 = get_idx_select_0(args['num_train'], args['frac_0'])
        eval_idx_select_0 = get_idx_select_0(args['num_eval'], args['frac_0'])
        if (verbose):
            print(f"train_idx_select_0: {train_idx_select_0}\neval_idx_select_0: {eval_idx_select_0}")
        for idx in train_idx_select_0:
            new_train_gold_idx_lst[idx] = 0
        for idx in eval_idx_select_0:
            new_eval_gold_idx_lst[idx] = 0
    if args['frac_top5'] > 0:
        train_idx_select_top5 = get_idx_select_top5(args['num_train'], args['frac_top5'], train_idx_select_0)
        eval_idx_select_top5 = get_idx_select_top5(args['num_eval'], args['frac_top5'], eval_idx_select_0)
        if (verbose):
            print(f"train_idx_select_top5: {train_idx_select_top5}\neval_idx_select_top5: {eval_idx_select_top5}")
        for idx in train_idx_select_top5:
            new_train_gold_idx_lst[idx] = random.choice([i for i in range(int(args['num_train'] * 0.05))])
        for idx in eval_idx_select_top5:
            new_eval_gold_idx_lst[idx] = random.choice([i for i in range(int(args['num_eval'] * 0.05))])
    if args['frac_0'] > 0 or args['frac_top5'] > 0:
        exper_train.initialize_gold_idx_lst(pre_gold_idx_lst=new_train_gold_idx_lst)
        exper_eval.initialize_gold_idx_lst(pre_gold_idx_lst=new_eval_gold_idx_lst)
        if (verbose):
            print(f"train_gold_idx_lst: {exper_train.gold_idx_lst}\neval_gold_idx_lst: {exper_eval.gold_idx_lst}")

    torch.save(exper_train, f"../dataset/pt/{filename}_seed{seed}_train_{args['num_train']}.pt")
    exper_train.generate_mistral_jsonl(f"{filename}_seed{seed}_train", "../dataset/jsonl/")

    torch.save(exper_eval, f"../dataset/pt/{filename}_seed{seed}_eval_{args['num_eval']}.pt")
    exper_eval.generate_mistral_jsonl(f"{filename}_seed{seed}_eval", "../dataset/jsonl/")

    if 'name_str' not in args.keys():
        with open(yaml_path, 'a') as fout:
            yaml.dump({'name_str':exper_train.get_str()}, fout, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml-path", 
        help="path to yaml file", 
        required=True
    )
    parser.add_argument(
        "--seed", 
        help="seed for randomness",
        type=int,
        default=0
    )
    parser.add_argument(
        "--verbose", 
        help="if verbose", 
        type=bool,
        default=True
    )
    parser.add_argument(
        "--only-check-task",
        help="if we only check the task", 
        type=bool,
        default=False
    )
    args = parser.parse_args()
    print(args.seed)
    main(
        args.yaml_path,
        args.seed,
        args.verbose,
        args.only_check_task
    )