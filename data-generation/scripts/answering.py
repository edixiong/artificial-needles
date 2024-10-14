import pathlib
import copy
from typing import List, Optional, Tuple
import random

def get_answer_dictionary(
        gold_key : tuple, gold_value, gold_dict_name : str, prompt_name : str, random_idx_lst=None
):
    ROOT = "../answers/dictionary/"

    if ("unshuffled" in prompt_name):
        assert(random_idx_lst == None)
    else:
        assert(random_idx_lst != None)

    answer_filename = f"{ROOT}{prompt_name}.answer"

    with open(answer_filename) as f:
        answer_template = f.read().rstrip("\n")


    # Configure gold_key_shuffle
    if ("unshuffled" in prompt_name):
        gold_key_str = str(gold_key)
        if ("unshuffled_singlesk" in prompt_name):
            gold_key_str = gold_key_str.replace("(", "")
            gold_key_str = gold_key_str.replace(",)", "")
        return answer_template.format(gold_key = gold_key_str,
                                      gold_value = str(gold_value),
                                      gold_dict_name = gold_dict_name)
    else:
        tmp_gold_key_lst_ref = list(gold_key)
        tmp_gold_key_lst = [tmp_gold_key_lst_ref[random_idx_lst[l]] for l in range(len(random_idx_lst))]
        tmp_gold_key_lst = copy.deepcopy(tmp_gold_key_lst)
        gold_key_shuffled = tuple(tmp_gold_key_lst)
        gold_key_shuffled_str = str(gold_key_shuffled)
        gold_key_shuffled_str = gold_key_shuffled_str.replace("(", "")
        gold_key_shuffled_str = gold_key_shuffled_str.replace(")", "")
        gold_key_str = str(gold_key)

        return answer_template.format(gold_key_shuffled=gold_key_shuffled_str, 
                                  gold_key = gold_key_str,
                                  gold_value = str(gold_value),
                                  gold_dict_name = gold_dict_name)