from dataclasses import dataclass, field
from typing import List, Dict

from dict_builder import build_dicts2
from prompting import get_prompt_dictionary
from answering import get_answer_dictionary
import copy
import random
import tiktoken

@dataclass
class RetrievalDataConfig:
    def valid_prob(self, prob : float) -> bool:
        if prob > 1.0 or prob < 0.0:
            return False
        else:
            return True
    
    def check_range(self, tup : tuple, tup_name : str):
        if len(tup) != 2:
            return False, f"{tup_name} is not a 2 tuple"
        elif tup[0] > tup[1]:
            return False, f"{tup_name} has first element greater than second element"
        else:
            return True, ""

@dataclass
class DictRetrievalDataConfig(RetrievalDataConfig):
    num_dicts : int 
    gold_dict_size : int
    dict_size_range : tuple
    gold_key_size : int
    key_size_range : tuple
    common_subkey_size : int
    prob_ng_has_common : float
    prob_ng_per_common : float
    key_common_ordered : bool  # Normally False
    gold_key_idx : int # if -1 then generate randomly; normally set it to be -1
    subkey_size_range : tuple
    val_size_range : tuple
    subkey_param : str
    val_param : str
    
    def __post_init__(self):
        tmp_bool, tmp_str = self.check_range(self.dict_size_range, "dict_size_range")
        if tmp_bool is False:
            raise ValueError(tmp_str)
        tmp_bool, tmp_str = self.check_range(self.key_size_range, "key_size_range")
        if tmp_bool is False:
            raise ValueError(tmp_str)
        tmp_bool, tmp_str = self.check_range(self.subkey_size_range, "subkey_size_range")
        if tmp_bool is False:
            raise ValueError(tmp_str)
        tmp_bool, tmp_str = self.check_range(self.val_size_range, "val_size_range")
        if tmp_bool is False:
            raise ValueError(tmp_str)
        if not (self.valid_prob(self.prob_ng_has_common)) or not (self.valid_prob(self.prob_ng_per_common)):
            raise ValueError("probability not valid")
        if (self.key_common_ordered and self.prob_ng_per_common != 1.0):
            raise ValueError("key_common_ordered is True but prob_ng_per_common is not 1.0")
        if (self.gold_key_idx < -1 or self.gold_key_idx >= self.gold_key_size):
            raise ValueError("gold_key_idx not valid")
        self.num_structs = self.num_dicts

@dataclass(init=False)
class SimpleDictRetrievalDataConfig(DictRetrievalDataConfig):
    def __init__(self, num_dicts, gold_dict_size, dict_size_range, gold_key_idx, 
                 subkey_size_range, val_size_range, subkey_param, val_param):
        super(SimpleDictRetrievalDataConfig, self).__init__(num_dicts, gold_dict_size, dict_size_range, 1, (1, 1), 0, 0, 0, False, 
                                                            gold_key_idx, subkey_size_range, val_size_range, subkey_param, val_param)
    def get_str(self):
        return f"ndicts{self.num_dicts}_gd{self.gold_dict_size}_drange{self.dict_size_range[0]}-{self.dict_size_range[1]}_skrange{self.subkey_size_range[0]}-{self.subkey_size_range[1]}_vrange{self.val_size_range[0]}-{self.val_size_range[1]}"

# ==========

@dataclass
class RetrievalTaskConfig:
    prompt_name : str
    max_token : int
    name_random : bool

@dataclass
class DictRetrievalTaskConfig(RetrievalTaskConfig):
    key_prompt_shuffled : bool
    
    def __post_init__(self):
        if ("_shuffled" not in self.prompt_name and self.key_prompt_shuffled is True):
            raise ValueError("Warning key_prompt_shuffled is True but prompt_name has no '_shuffled'")
        if ("_shuffled" in self.prompt_name and self.key_prompt_shuffled is False):
            raise ValueError("Warning key_prompt_shuffled is False but prompt_name has '_shuffled'")

@dataclass(init=False)
class SimpleDictRetrievalTaskConfig(DictRetrievalTaskConfig):
    def __init__(self, prompt_name, max_token, name_random):
        super(SimpleDictRetrievalTaskConfig, self).__init__(prompt_name, max_token, name_random, False)
        if "singlesk" not in prompt_name:
            raise ValueError("\"singlesk\" should be in the prompt_name")
    def get_str(self):
        return f"pname-{self.prompt_name}_nran-{self.name_random}"