# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pkl

import argparse
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append("..")
from common_utils.utils import pkl_dump,pkl_load
from sklearn.model_selection import train_test_split

def load_data(case_control_path,case_control_filename, features_path,features_filename,data_filename):
    ptIDs = pd.read_csv(f"{case_control_path}{case_control_filename}.csv",usecols=['PATID'],dtype =str)
    fea2id, features = pkl_load(f"{features_filename}.pkl", features_path)
    data = pkl_load(f"{data_filename}.pkl", features_path)
    return ptIDs,fea2id,data

def to_matrix(all_data, col_num, num_fea_cols):
    pids = []
    matrix = []

    for idx, d in enumerate(all_data):
        m = np.zeros(col_num + 1)
        if num_fea_cols == -1:
            d1 = d
        else:
            d1 = d[:-num_fea_cols]
            
        for index, feature_id in enumerate(d1):
            if index == 0:
                m[0] = feature_id
            else:
                m[feature_id] = 1
        pids.append(idx)
        matrix.append(m)
    return matrix, pids

def create_data(matrix):
    np.random.seed(13)
    np.random.shuffle(matrix)
    np.random.seed(47)
    np.random.shuffle(matrix)
    dx = []
    dy = []
    for each in matrix:
        dx.append(each[1:])
        dy.append(each[0])
    dx = np.array(dx)
    dy = np.array(dy)
    return dx, dy

def main(args):

    case_control_filename = args.case_control_filename
    features_filename = args.features_filename
    data_filename = args.data_filename
    case_control_path = args.case_control_path
    features_path = args.features_path
    output_path = args.output_path

    ptIDs,fea2id,data = load_data(case_control_path,features_path,case_control_filename,features_filename,data_filename)

    for i in tqdm(range(5)):
        train_id, test_id = train_test_split(ptIDs,test_size=0.2)
        test_ids = test_id.PATID.to_list()
        train_ids = train_id.PATID.to_list()

        trains = []
        tests = []
        count = 0
        for dp in data:
            pid = dp[0]
            ndata = dp[1:]
            if pid in train_ids:
                trains.append(ndata)
            elif pid in test_ids:
                tests.append(ndata)
            else:
                count = count + 1

        matrix, pids = to_matrix(trains, len(fea2id), -1)
        tr_dx, tr_dy = create_data(matrix)
        matrix, pids = to_matrix(tests, len(fea2id), -1)
        ts_dx, ts_dy = create_data(matrix)
        pkl_dump((tr_dx, tr_dy, ts_dx, ts_dy), "expr_data_CC0yr_train_test.pkl", output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # case_control_path
    parser.add_argument("--case_control_path", required=True, type=str, 
        help="the case control patients that will be processed")

    # case_control_filename
    parser.add_argument("--case_control_filename", required=True, type=str, 
        help="the case control filename")

    # features_path
    parser.add_argument("--features_path", required=True, type=str, 
        help="the features that will be processed")

    # features_filename
    parser.add_argument("--features_filename", required=True, type=str, 
        help="the features filename")

    # data_filename
    parser.add_argument("--data_filename", required=True, type=str, 
        help="the data filename")

    # output_path
    parser.add_argument("--output_path", required=True, type=str, 
        help="the output_path")