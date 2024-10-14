from typing import List

from dict_builder import build_dicts2
from message_builder import build_msgs
from prompting import get_prompt_dictionary
from answering import get_answer_dictionary
from config import RetrievalDataConfig, RetrievalTaskConfig, SimpleDictRetrievalDataConfig

from transformers import LlamaTokenizer, AutoTokenizer
import copy
import random
import torch
import json
import tiktoken

HF_TOKEN = "<your-hf-token>"

class RetrievalTask:
    def __init__(self, dataconfig : RetrievalDataConfig, taskconfig : RetrievalTaskConfig, tokenizer_name : str):
        self.dataconfig = dataconfig
        self.taskconfig = taskconfig
        self.tokenizer_name = tokenizer_name
        if tokenizer_name == "tiktoken":
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif tokenizer_name == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat', token=HF_TOKEN)
        else:
            raise ValueError("tokenizer name invalid")
        self.build_data()

    def get_prompt_len(self, prompt):
        if self.tokenizer_name == "tiktoken":
            return len(self.tokenizer.encode(prompt))
        elif self.tokenizer_name == "llama":
            return len(self.tokenizer(prompt)['input_ids'])
        else:
            raise RuntimeError("tokenizer not expected")
    def build_data(self):
        raise NotImplementedError


class DictRetrieval(RetrievalTask):
    def __init__(self, dataconfig, taskconfig, tokenizer_name):
        super(DictRetrieval, self).__init__(dataconfig, taskconfig, tokenizer_name)
        
    def build_data(self):
        exceed = True
        i = 0
        while (exceed and i < 1000):
            gold_dict, ng_dict_lst, gold_key, gold_val, common_subkey_lst, gold_key_idx = build_dicts2(self.dataconfig.num_dicts, self.dataconfig.gold_dict_size, self.dataconfig.dict_size_range, self.dataconfig.gold_key_size, 
                                                                                                           self.dataconfig.key_size_range, self.dataconfig.common_subkey_size, self.dataconfig.prob_ng_has_common, self.dataconfig.prob_ng_per_common, 
                                                                                                           self.dataconfig.key_common_ordered, self.dataconfig.gold_key_idx, self.dataconfig.subkey_size_range, self.dataconfig.val_size_range, 
                                                                                                           self.dataconfig.subkey_param, self.dataconfig.val_param)
            if self.within_maxtoken(gold_dict, ng_dict_lst, gold_key):
                exceed = False
            i += 1
        if exceed is False:
            self.gold_dict = gold_dict
            self.ng_dict_lst = ng_dict_lst
            self.gold_key = gold_key
            self.gold_val = gold_val
            self.common_subkey_lst = common_subkey_lst
            self.gold_key_idx = gold_key_idx
        else:
            raise ValueError("fails to build data as dataconfig and taskconfig mismatches (too many data for max_token); please adjust dataconfig or taskconfig")
        
        if (self.taskconfig.name_random == False):
            name_lst = [(it + 1) for it in range(self.dataconfig.num_dicts)]
        else:
            name_lst = [(it + 1) for it in range(self.dataconfig.num_dicts)]
            random.shuffle(name_lst)
        self.name_lst = copy.deepcopy(name_lst)

        tmp_random_idx_lst = [it for it in range(len(gold_key))]
        random.shuffle(tmp_random_idx_lst)
        self.tmp_random_idx_lst = tmp_random_idx_lst
    
    def within_maxtoken(self, gold_dict, ng_dict_lst, gold_key, redundancy=100) -> bool:
        tmp_dict_lst = copy.deepcopy(ng_dict_lst)
        tmp_dict_lst.insert(0, copy.deepcopy(gold_dict))
        name_lst_tmp = [i+1 for i in range(len(tmp_dict_lst))]
        if self.taskconfig.key_prompt_shuffled:
            ridx_lst = [it for it in range(len(gold_key))]
            random.shuffle(ridx_lst)
            tmp_prompt = get_prompt_dictionary(tmp_dict_lst, copy.deepcopy(gold_key), self.taskconfig.prompt_name, 
                                               name_lst_tmp, copy.deepcopy(ridx_lst))
        else:
            tmp_prompt = get_prompt_dictionary(tmp_dict_lst, copy.deepcopy(gold_key), self.taskconfig.prompt_name, 
                                               name_lst_tmp, None)
        num_tokens = self.get_prompt_len(tmp_prompt)
        return (num_tokens < self.taskconfig.max_token - redundancy)
    
    def get_prompt(self, gold_dict_idx : int):
        assert (gold_dict_idx >= 0 and gold_dict_idx <= self.dataconfig.num_dicts)
        dict_lst = copy.deepcopy(self.ng_dict_lst)
        dict_lst.insert(gold_dict_idx, copy.deepcopy(self.gold_dict))
        if self.taskconfig.key_prompt_shuffled:
            prompt = get_prompt_dictionary(dict_lst, copy.deepcopy(self.gold_key), self.taskconfig.prompt_name, 
                                           copy.deepcopy(self.name_lst), copy.deepcopy(self.tmp_random_idx_lst))
        else:
            prompt = get_prompt_dictionary(dict_lst, copy.deepcopy(self.gold_key), self.taskconfig.prompt_name, 
                                           copy.deepcopy(self.name_lst), None)
        num_tokens = self.get_prompt_len(prompt)
        if (num_tokens < self.taskconfig.max_token - 30):
            return prompt
        else:
            raise ValueError("fails to get prompt -- please adjust dataconfig or taskconfig")
    
    def get_answer(self, gold_dict_idx : int):
        assert (gold_dict_idx >= 0 and gold_dict_idx < self.dataconfig.num_dicts)
        dict_lst = copy.deepcopy(self.ng_dict_lst)
        dict_lst.insert(gold_dict_idx, copy.deepcopy(self.gold_dict))
        gold_dict_name = self.name_lst[gold_dict_idx]
        if self.taskconfig.key_prompt_shuffled:
            answer = get_answer_dictionary(copy.deepcopy(self.gold_key), self.gold_val, gold_dict_name, 
                                           self.taskconfig.prompt_name, copy.deepcopy(self.tmp_random_idx_lst))
        else:
            answer = get_answer_dictionary(copy.deepcopy(self.gold_key), self.gold_val, gold_dict_name, 
                                           self.taskconfig.prompt_name, None)
        return answer
        
    def change_prompt_name(self, new_prompt_name):
        self.taskconfig.prompt_name = new_prompt_name
        if ("_unshuffled" in new_prompt_name):
            self.taskconfig.key_prompt_shuffled = False
        elif ("_shuffled" in new_prompt_name):
            self.taskconfig.key_prompt_shuffled = True
        else:
            raise ValueError("either '_unshuffled' or '_shuffled' is found in new_prompt_name")
    
    def get_openai_entry(self, gold_dict_idx : int):
        entry = {}
        msg_1 = {"role": "system", "content": "Write a response that appropriately completes the request."}
        msg_2 = {"role": "user", "content": self.get_prompt(gold_dict_idx)}
        msg_3 = {"role": "assistant", "content": self.get_answer(gold_dict_idx)}
        messages_lst = [msg_1, msg_2, msg_3]
        entry["messages"] =  messages_lst
        return entry
    
    def get_mistral_entry(self, gold_dict_idx : int):
        entry = {}
        msg_2 = {"role": "user", "content": self.get_prompt(gold_dict_idx)}
        msg_3 = {"role": "assistant", "content": self.get_answer(gold_dict_idx)}
        messages_lst = [msg_2, msg_3]
        entry["messages"] =  messages_lst
        return entry

class SimpleRetrieval(RetrievalTask):
    def __init__(self, dataconfig, taskconfig, tokenizer_name):
        super(SimpleRetrieval, self).__init__(dataconfig, taskconfig, tokenizer_name)
        
    def build_data(self):
        exceed = True
        i = 0
        while (exceed and i < 1000):
            gold_msg, ng_msg_lst = build_msgs(self.dataconfig.num_msgs, self.dataconfig.content_param, self.dataconfig.content_range)
            if self.within_maxtoken(gold_msg, ng_msg_lst):
                exceed = False
            i += 1
        if exceed is False:
            self.gold_msg = gold_msg
            self.ng_msg_lst = ng_msg_lst
        else:
            raise ValueError("fails to build data as dataconfig and taskconfig mismatches (too many data for max_token); please adjust dataconfig or taskconfig")
        
        if (self.taskconfig.name_random == False):
            name_lst = [(it + 1) for it in range(self.dataconfig.num_msgs)]
        else:
            name_lst = [(it + 1) for it in range(self.dataconfig.num_msgs)]
            random.shuffle(name_lst)
        self.name_lst = copy.deepcopy(name_lst)
    
    def within_maxtoken(self, gold_msg, ng_msg_lst, redundancy=100) -> bool:
        tmp_msg_lst = copy.deepcopy(ng_msg_lst)
        tmp_msg_lst.insert(0, copy.deepcopy(gold_msg))
        name_lst_tmp = [i+1 for i in range(len(tmp_msg_lst))]
        tmp_prompt = get_prompt_simple(tmp_msg_lst, name_lst_tmp[0], self.taskconfig.prompt_name, name_lst_tmp)
        num_tokens = self.get_prompt_len(tmp_prompt)
        return (num_tokens < self.taskconfig.max_token - redundancy)
    
    def get_prompt(self, gold_msg_idx : int):
        assert (gold_msg_idx >= 0 and gold_msg_idx < self.dataconfig.num_msgs)
        msg_lst = copy.deepcopy(self.ng_msg_lst)
        msg_lst.insert(gold_msg_idx, copy.deepcopy(self.gold_msg))
        prompt = get_prompt_simple(msg_lst, self.name_lst[gold_msg_idx], self.taskconfig.prompt_name, self.name_lst)
        num_tokens = self.get_prompt_len(prompt)
        if (num_tokens < self.taskconfig.max_token - 30):
            return prompt
        else:
            raise ValueError("fails to get prompt -- please adjust dataconfig or taskconfig")
    
    def get_answer(self, gold_msg_idx : int):
        assert (gold_msg_idx >= 0 and gold_msg_idx < self.dataconfig.num_msgs)
        msg_lst = copy.deepcopy(self.ng_msg_lst)
        msg_lst.insert(gold_msg_idx, copy.deepcopy(self.gold_msg))
        answer = get_answer_simple(self.gold_msg, self.name_lst[gold_msg_idx], self.taskconfig.prompt_name)
        return answer
    
    def change_prompt_name(self, new_prompt_name):
        self.taskconfig.prompt_name = new_prompt_name
    
    def get_openai_entry(self, gold_tree_idx : int):
        entry = {}
        msg_1 = {"role": "system", "content": "Write a response that appropriately completes the request."}
        msg_2 = {"role": "user", "content": self.get_prompt(gold_tree_idx)}
        msg_3 = {"role": "assistant", "content": self.get_answer(gold_tree_idx)}
        messages_lst = [msg_1, msg_2, msg_3]
        entry["messages"] =  messages_lst
        return entry

# ========================

class Exper:
    def __init__(self, num_experiment : int, tokenizer, dataconfig=None, taskconfig=None, task_lst=None):
        self.num_experiment = num_experiment
        self.tokenizer = tokenizer
        if (num_experiment <= 0):
            raise ValueError("num_experiment cannot be <= 0")
        if (task_lst != None):
            if ((dataconfig != None) or (taskconfig != None)):
                raise ValueError("task_lst not None but at least one of dataconfig and taskconfig is not None")
            if (num_experiment != len(task_lst)):
                raise ValueError("length of task_lst is not equal to num_experiment")
            self.task_lst = task_lst
        else:
            if ((dataconfig is None) or (taskconfig is None)):
                raise ValueError("task_lst is None and at least one of dataconfig and taskconfig is None")
            self.task_lst = None
            self.dataconfig = dataconfig
            self.taskconfig = taskconfig

    def get_tasks(self):
        return copy.deepcopy(self.task_lst)
    
    def initialize_gold_idx_lst(self, mode="uniform", pre_gold_idx_lst=None):
        if (pre_gold_idx_lst != None):
            # verify gold_idx_lst
            if len(pre_gold_idx_lst) != self.num_experiment:
                raise ValueError("pre_gold_idx_lst has length different from self.num_experiment")
            for i in range(self.num_experiment):
                if (pre_gold_idx_lst[i] < 0 or pre_gold_idx_lst[i] >= self.task_lst[i].dataconfig.num_structs):
                    raise ValueError(f"pre_gold_idx_lst[{i}] is not valid")
            self.gold_idx_lst = copy.deepcopy(pre_gold_idx_lst)
        else:
            gold_idx_lst = []
            if mode == "uniform":
                print("initializing uniformly")
                for i in range(self.num_experiment):
                    gold_idx_lst.append(random.randint(0, self.task_lst[i].dataconfig.num_structs - 1))
            else:
                raise ValueError("other mode not supported yet")
            self.gold_idx_lst = gold_idx_lst
                

    def generate_openai_jsonl(self, filename : str):
        if self.gold_idx_lst == None:
            raise RuntimeError("self.gold_idx_lst not initialized yet")
        if ("." in filename):
            print("Warning: '.' is in filename")
        name = f"{filename}_{self.num_experiment}.jsonl"
        with open(name, 'w') as outfile:
            for i in range(self.num_experiment):
                entry = self.task_lst[i].get_openai_entry(self.gold_idx_lst[i])
                json.dump(entry, outfile)
                outfile.write("\n")
    
    def generate_mistral_jsonl(self, filename : str, dir="./"):
        if self.gold_idx_lst == None:
            raise RuntimeError("self.gold_idx_lst not initialized yet")
        if ("." in filename):
            print("Warning: '.' is in filename")
        name = f"{dir}{filename}_{self.num_experiment}.jsonl"
        with open(name, 'w') as outfile:
            for i in range(self.num_experiment):
                entry = self.task_lst[i].get_mistral_entry(self.gold_idx_lst[i])
                json.dump(entry, outfile)
                outfile.write("\n")

class DictRetrievalExper(Exper):
    def __init__(self, num_experiment, tokenizer, dataconfig=None, taskconfig=None, task_lst=None):
        super(DictRetrievalExper, self).__init__(num_experiment, tokenizer, dataconfig, taskconfig, task_lst)
        if (self.task_lst is None):
            self.task_lst = [DictRetrieval(copy.deepcopy(dataconfig), copy.deepcopy(taskconfig), tokenizer) for i in range(num_experiment)]
        else:
            assert((dataconfig is None) and (taskconfig is None))
            self.task_lst = task_lst
        self.initialize_gold_idx_lst(mode="uniform")
    def __str__(self):
        if self.dataconfig is None:
            return "DictRrtrievalExper"
        return f"DictRetrievalExper(\n\tnum_experiment={self.num_experiment},\n\tdataconfig={str(self.dataconfig)},\n\ttaskconfig={str(self.taskconfig)}\n)"
    def get_str(self):
        if self.dataconfig == None or self.taskconfig == None:
            raise NotImplementedError
        if isinstance(self.dataconfig, SimpleDictRetrievalDataConfig):
            return f"simpledict_{self.dataconfig.get_str()}_{self.taskconfig.get_str()}"
        else:
            raise NotImplementedError


class SimpleRetrievalExper(Exper):
    def __init__(self, num_experiment, tokenizer, dataconfig=None, taskconfig=None, task_lst=None):
        super(SimpleRetrievalExper, self).__init__(num_experiment, tokenizer, dataconfig, taskconfig, task_lst)
        if (self.task_lst is None):
            self.task_lst = [SimpleRetrieval(copy.deepcopy(dataconfig), copy.deepcopy(taskconfig), tokenizer) for i in range(num_experiment)]
        else:
            assert((dataconfig is None) and (taskconfig is None))
            self.task_lst = task_lst
        self.initialize_gold_idx_lst()
    def __str__(self):
        if self.dataconfig is None:
            return "SimpleRetrievalExper"
        return f"SimpleRetrievalExper(\n\tnum_experiment={self.num_experiment},\n\tdataconfig={str(self.dataconfig)},\n\ttaskconfig={str(self.taskconfig)}\n)"