import torch
from torch import tensor

import math
from typing import NewType, List, Tuple, Union

Bitword = NewType('Bitword', List[int])
Batch = NewType("Batch", Tuple[torch.LongTensor, torch.LongTensor])

def zip_(a, b, offset=1):
    return zip(a, b[offset:])

def pad_chunk(a, size):
    a =[0] * (math.ceil(len(a) / size) * size - len(a)) + a
    return [a[i:i+size] for i in range(0, len(a), size)]

def bw2i(lst):
    try:
        return int("".join(map(str, lst)), 2)
    except:
        raise ValueError("Expected a List")
    
def bw2oh(lst, width):
    res = torch.zeros(2 ** width) # naming convention 'width' used in the paper
    idx = bw2i(lst)
    res[idx] = 1.0  
    return res

def i2bw_str(label, width):
    return bin(label)[2:].rjust(width, '0')

def i2bw(label, width=None):
    if width is None:
        width = int(math.log2(label)) + 1
    return [int(i) for i in i2bw_str(label, width)]

def bw2str(lst):
    return i2bw_str(bw2i(lst), len(lst))

def char2bw(char, chars, width):
    idx = chars.index(char)
    char_bw = f"{idx:b}".rjust(width, "0")
    return [int(i) for i in char_bw[-width:]]

def bw2char(bw, chars):
    return chars[bw2i(bw)]
