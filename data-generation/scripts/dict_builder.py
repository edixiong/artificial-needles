import random   
from typing import List, Tuple
import copy

import os

def gold_unique(dict_lst, gold_key, gold_val) -> bool:
    unique = False
    for d in dict_lst:
        for k, v in d.items():
            if unique == False:
                if k == gold_key and v == gold_val:
                    unique = True
            elif unique == True and (k == gold_key or v == gold_val):
                unique = False
                break
    return unique

def generate_val(val_size_range, val_param):
    val_size = random.randint(val_size_range[0], val_size_range[1])
    if (val_param == "numerical"):
        return random.randint(10**(val_size - 1), 10**val_size - 1)
    else:
        raise ValueError("subkey_param invalid.")

def generate_subkey(subkey_size_range, subkey_param):
    subkey_size = random.randint(subkey_size_range[0], subkey_size_range[1])
    if (subkey_param == "numerical"):
        return random.randint(10**(subkey_size - 1), 10**subkey_size - 1)
    elif (subkey_param == "vocab"):
        random_word = r.get_random_word()
        return random_word
    elif (subkey_param == "string"):
        return "".join(random.choices(alphabet_lst, k=subkey_size))
    else:
        raise ValueError("subkey_param invalid.")
        
def generate_key_lst(key_size : int, common_subkey_lst : list, prob_has_common : float, prob_per_common :float, 
                     key_common_ordered : bool, subkey_size_range : tuple, subkey_param : str) -> list:
    if (prob_per_common < 1 and key_common_ordered == True):
        raise ValueError("if key_common_ordered equals True then prob_per_common must be true")
    subkey_lst_tmp = []
    rand1 = random.random()
    if (rand1 < prob_has_common):
        if key_common_ordered:
            for i in range(key_size - len(common_subkey_lst)):
                subkey_lst_tmp.append(generate_subkey(subkey_size_range, subkey_param))
            random_idx = random.randint(0, key_size - len(common_subkey_lst))
            subkey_lst = subkey_lst_tmp[0:random_idx] + common_subkey_lst + subkey_lst_tmp[random_idx:]
            # assert(len(subkey_lst) == key_size)
            return subkey_lst
        else:
            for j in range(len(common_subkey_lst)):
                rand2 = random.random()
                if (rand2 < prob_per_common):
                    random_idx = random.randint(0, len(subkey_lst_tmp))
                    subkey_lst_tmp.insert(random_idx, common_subkey_lst[j])
            for k in range(key_size - len(subkey_lst_tmp)):
                random_idx = random.randint(0, len(subkey_lst_tmp))
                subkey_lst_tmp.insert(random_idx, generate_subkey(subkey_size_range, subkey_param))
            # assert(len(subkey_lst_tmp) == key_size)
            return subkey_lst_tmp
    else:
        for i in range(key_size):
            subkey_lst_tmp.append(generate_subkey(subkey_size_range, subkey_param))
        # assert(len(subkey_lst_tmp) == key_size)
        return subkey_lst_tmp


def generate_not_gold_kv_pair(gold_key, gold_val, key_size, common_subkey_lst, prob_ng_has_common, prob_ng_per_common, 
                              key_common_ordered, subkey_size_range, val_size_range, subkey_param, val_param):
    key = tuple(generate_key_lst(key_size, common_subkey_lst, prob_ng_has_common, prob_ng_per_common, 
                                 key_common_ordered, subkey_size_range, subkey_param))
    val = generate_val(val_size_range, val_param)
    while ((key == gold_key) or (val == gold_val)): # if same as gold_key or gold_val, regenerate
        key = tuple(generate_key_lst(key_size, common_subkey_lst, prob_ng_has_common, prob_ng_per_common, 
                                     key_common_ordered, subkey_size_range, subkey_param))
        val = generate_val(val_size_range, val_param)
    return key, val

def generate_dict_not_gold(dict_size : int, key_size_range : tuple, common_subkey_lst : list, 
                           prob_ng_has_common : float, prob_ng_per_common : float, key_common_ordered : bool, subkey_size_range : tuple,
                            val_size_range : tuple, subkey_param : str, val_param : str, gold_key : list, gold_val) -> dict:
    dict_not_gold = {}
    for i in range(dict_size):
        tmp_key_size = random.randint(key_size_range[0], key_size_range[1])
        key, val = generate_not_gold_kv_pair(gold_key, gold_val, tmp_key_size, common_subkey_lst, prob_ng_has_common, prob_ng_per_common,
                                             key_common_ordered, subkey_size_range, val_size_range, subkey_param, val_param)
        dict_not_gold[key] = val
    return dict_not_gold

def build_dicts(num_dicts : int, gold_dict_size : int, dict_size_range : tuple, gold_key_size : int, key_size_range : tuple, 
                common_subkey_size : int, prob_ng_has_common : float, prob_ng_per_common : float, key_common_ordered, gold_dict_idx=-1, gold_key_idx = -1, 
                subkey_size_range = (3, 3), val_size_range = (4, 4), subkey_param="numerical", val_param = "numerical") -> List[dict]:
    assert(gold_key_size > common_subkey_size)
    assert(key_size_range[0] > common_subkey_size)
    # if gold_dict_idx == -1, randomly generate one
    if (gold_dict_idx == -1):
        gold_dict_idx = random.randint(0, num_dicts - 1)
    # if gold_key_idx == -1, randomly generate one
    if (gold_key_idx == -1):
        gold_key_idx = random.randint(0, gold_dict_size - 1)
    # generate common_subkey_lst
    common_subkey_lst = []
    for i in range(common_subkey_size):
        common_subkey_lst.append(generate_subkey(subkey_size_range, subkey_param))
    # generate gold dict
    gold_key = tuple(generate_key_lst(gold_key_size, common_subkey_lst, 1.0, 1.0, key_common_ordered, subkey_size_range, subkey_param))
    gold_val = generate_val(val_size_range, val_param)
    gold_dict = {}
    for j in range(gold_dict_size):
        if j == gold_key_idx:
            gold_dict[gold_key] = gold_val
        else:
            tmp_key_size = random.randint(key_size_range[0], key_size_range[1])
            key, val = generate_not_gold_kv_pair(gold_key, gold_val, tmp_key_size, common_subkey_lst, prob_ng_has_common, prob_ng_per_common,
                                                 key_common_ordered, subkey_size_range, val_size_range, subkey_param, val_param)
            gold_dict[key] = val
    # build dict_lst
    dict_lst = []
    for k in range(num_dicts):
        if k == gold_dict_idx:
            dict_lst.append(gold_dict)
        else:
            tmp_dict_size = random.randint(dict_size_range[0], dict_size_range[1])
            dict_lst.append(generate_dict_not_gold(tmp_dict_size, key_size_range, common_subkey_lst, prob_ng_has_common, prob_ng_per_common, 
                                                   key_common_ordered, subkey_size_range, val_size_range, subkey_param, val_param, gold_key, gold_val))
    return dict_lst, gold_key, gold_val, common_subkey_lst, gold_dict_idx, gold_key_idx


def build_dicts2(num_dicts : int, gold_dict_size : int, dict_size_range : tuple, gold_key_size : int, key_size_range : tuple, 
                common_subkey_size : int, prob_ng_has_common : float, prob_ng_per_common : float, key_common_ordered, gold_key_idx = -1, 
                subkey_size_range = (3, 3), val_size_range = (4, 4), subkey_param="numerical", val_param = "numerical") -> List[dict]:
    
    dict_lst, gold_key, gold_val, common_subkey_lst, gold_dict_idx, gold_key_idx = build_dicts(num_dicts, gold_dict_size, dict_size_range, gold_key_size, key_size_range, 
                common_subkey_size, prob_ng_has_common, prob_ng_per_common, key_common_ordered, 0, gold_key_idx, 
                subkey_size_range, val_size_range, subkey_param, val_param)
    
    assert(gold_dict_idx == 0)
    gold_dict = dict_lst.pop(0)
    ng_dict_lst = copy.deepcopy(dict_lst)
    assert(len(ng_dict_lst) == num_dicts - 1)

    return gold_dict, ng_dict_lst, gold_key, gold_val, common_subkey_lst, gold_key_idx