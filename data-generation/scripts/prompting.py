import copy
from typing import List, Dict, Tuple
import random

def get_prompt_dictionary(
    dict_list : List[dict], gold_key : tuple, prompt_name : str, name_lst : List[str], random_idx_lst=None
):  
    ROOT = "../prompts/dictionary/"
    
    if ("unshuffled" in prompt_name):
        assert(random_idx_lst == None)
    else:
        assert (random_idx_lst != None)

    prompt_filename = f"{ROOT}{prompt_name}.prompt"

    with open(prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")

    # Configure gold_key_shuffle
    if ("unshuffled" in prompt_name):
        gold_key_str = str(gold_key)
        if ("unshuffled_singlesk" in prompt_name):
            gold_key_str = gold_key_str.replace("(", "")
            gold_key_str = gold_key_str.replace(",)", "")
    else:
        tmp_gold_key_lst_ref = list(gold_key)
        tmp_gold_key_lst = [tmp_gold_key_lst_ref[random_idx_lst[l]] for l in range(len(random_idx_lst))]
        tmp_gold_key_lst = copy.deepcopy(tmp_gold_key_lst)
        gold_key_shuffled = tuple(tmp_gold_key_lst)
        gold_key_str = str(gold_key_shuffled)
        gold_key_str = gold_key_str.replace("(", "")
        gold_key_str = gold_key_str.replace(")", "")

    # Configure formatted disctionaries
    formatted_dictionaries = []
    for dict_idx, d in enumerate(dict_list):
        str_d = str(d)
        if ("unshuffled_singlesk" in prompt_name):
            str_d = str_d.replace("(", "")
            str_d = str_d.replace(",)", "")
        formatted_dictionaries.append(f"Dictionary [{name_lst[dict_idx]}] {str_d}")
    return prompt_template.format(gold_key_str=gold_key_str, disctionaries="\n".join(formatted_dictionaries))
