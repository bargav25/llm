from collections import Counter 
from typing import List, Tuple 

import math
import torch


def merge(token: List[int], pair: Tuple[int, int], new_idx: int) -> List[int]:
    
    res, i = [], 0

    while i < len(token):
        if i+1 < len(token) and (token[i], token[i+1]) == pair:
            res.append(new_idx)
            i += 1
        else:
            res.append(token[i])
        i+=1

    return res


def get_best_pair(pair_counts: Counter):
    return max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]



