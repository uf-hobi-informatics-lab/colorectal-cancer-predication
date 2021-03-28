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

def load_data(features_path,features_filename,data_filename):
    fea2id = pkl_load(f"{features_filename}", features_path)
    data = pkl_load(f"{data_filename}", features_path)
    return fea2id,data

def convert_to_dataframe(fea2id,patient_data):
    patient_with_features = pd.DataFrame(columns = ['pid']+list(fea2id.values()))

    for i in range(len(patient_data)):
        patient_with_features.loc[i,'pid']=patient_data[i][0]
        for j in range(1, len(fea2id)+1):
            if j in patient_data[i][1:]:
                patient_with_features.loc[i,j]=1
            else:
                patient_with_features.loc[i,j]=0
    return patient_with_features

def main(args):

    features_path = args.features_path
    features_filename = args.features_filename
    data_filename = args.data_filename
    output_path = args.output_path

    fea2id,patient_data = load_data(features_path,features_filename,data_filename)
    print('Load Data Done!')
    patient_with_features = convert_to_dataframe(fea2id,patient_data)
    print('Finished!')
    patient_with_features.to_csv(f"{output_path}/{data_filename}.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    main(args)