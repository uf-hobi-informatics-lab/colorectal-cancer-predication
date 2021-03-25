# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle as pkl

def pkl_dump(data, file):
    with open(file, "wb") as fw:
        pkl.dump(data, fw)

        
def pkl_load(file):
    with open(file, "rb") as fr:
        data = pkl.load(fr)
    return data