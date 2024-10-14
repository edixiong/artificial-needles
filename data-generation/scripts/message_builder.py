import random   
from typing import List, Tuple
import copy

def generate_msg(content_param, content_range):
    return random.randint(0, 10000)

def build_msgs(num_msgs : int, content_param : str, content_range : tuple):
    if (num_msgs < 2):
        raise ValueError("num_msg must be >=2.")
    return generate_msg(content_param, content_range), [generate_msg(content_param, content_range) for i in range(num_msgs - 1)]