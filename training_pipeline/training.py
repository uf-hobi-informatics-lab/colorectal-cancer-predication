# -*- coding: utf-8 -*-
import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='(%(asctime)s) [%(process)d] %(levelname)s: %(message)s')

import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime
from utils import pkl_dump,pkl_load
from config import ModelType

def load_data(case_control_path,encoding_path,case_control_filename,encoding_filename,data_filename):
    ptIDs = pd.read_csv(f"{case_control_path}{case_control_filename}.csv",usecols=['PATID'],dtype =str)
    fea2id, features = pkl_load(f"{encoding_path}{encoding_filename}.pkl")
    data = pkl_load(f"{case_control_filename}{data_filename}.pkl")

    return ptIDs,fea2id,data


def to_matrix(all_data, col_num, num_fea_cols):
    pids = []
    matrix = []
    for idx, d in enumerate(all_data):
        m = np.zeros(col_num + 1)
        mp = []
        if num_fea_cols == -1:
            d1 = d
        else:
            d1 = d[:-num_fea_cols]
        for i, e in enumerate(d1):
            if i == 0:
                m[0] = e
            else:
                m[e] = 1.
        
        if num_fea_cols == -1:
            d2 = []
        else:
            d2 = d[-num_fea_cols:]
        for e in d2:
            mp.append(e)
        
        pids.append(idx)
        nmn = np.concatenate((m, np.array(mp)))
        matrix.append(nmn)
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
    print(dx.shape, dy.shape)
    return dx, dy


def run_experiment(clf, params, tasks, nb, nit, model_type):
    for task in tasks:
        print(f"current task: {task} {model_name}")
        model_dump = f"{task}year_{model_name}_model.pkl"

        tr_dx, tr_dy, ts_dx, ts_dy = pkl_load(f"expr_data_CC{task}yr_train_test.pkl")
        
        cv_model = RandomizedSearchCV(clf, params, scoring='roc_auc', n_jobs=nb, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=13), 
                                      verbose=1, n_iter=nit, random_state=13)
        cv_model.fit(tr_dx, tr_dy)
        opt_clf = cv_model.best_estimator_
        pkl_dump(opt_clf, model_dump)

        preds = opt_clf.predict_proba(ts_dx)
        pkl_dump(preds, f"{task}year_{model_name}_preds.pkl")

        idx = np.argmax(opt_clf.classes_)
        preds_1 = list(map(lambda x: x[idx], preds))

        auc_score = roc_auc_score(ts_dy, preds_1)
        fprs, tprs, ths = roc_curve(ts_dy, preds_1)
        print("auc_score is : ",auc_score)
        
        J_idx = np.argmax(tprs - fprs)
        fpr, tpr, th = fprs[J_idx], tprs[J_idx], ths[J_idx]
        auc_score1 = auc(fprs, tprs)

        sen = tpr
        spe = 1 - fpr
        stats = [sen, spe, auc_score1]

def main(args):

    case_control_filename = args.case_control_filename
    encoding_filename = args.encoding_filename
    data_filename = args.data_filename
    case_control_path = args.case_control_path
    encoding_path = args.encoding_path
    model_type = args.model_type
    tuned_parameters = args.tuned_parameters

    if model_type is ModelType.M_LR:
        clf = LogisticRegression(warm_start=True)
    elif model_type is ModelType.M_RF:
        clf = RandomForestClassifier()
    elif model_type is ModelType.M_SVM:
        clf = svm.SVC(probability=True)
    else:
        clf = lgb.LGBMClassifier(objective='binary', 
                         n_jobs=10, 
                         scale_pos_Weight=4, use_missing=True)

    ptIDs,fea2id,data = load_data(case_control_path,encoding_path,case_control_filename,encoding_filename,data_filename)

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
        pkl_dump((tr_dx, tr_dy, ts_dx, ts_dy), "expr_data_CC0yr_train_test.pkl")

        run_experiment(clf, tuned_parameters, tasks, nb, nit, model_type)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # case_control_path
    parser.add_argument("--case_control_path", required=True, type=str, 
        help="the case control patients that will be processed")

    # encoding_path
    parser.add_argument("--encoding_path", required=True, type=str, 
        help="the encoding features that will be processed")

    # case_control_filename
    parser.add_argument("--case_control_filename", required=True, type=str, 
        help="the case control filename")

    # encoding_filename
    parser.add_argument("--encoding_filename", required=True, type=str, 
        help="the features filename")

    # data_filename
    parser.add_argument("--data_filename", required=True, type=str, 
        help="the data filename")

    # model_type
    parser.add_argument("--model_type", required=True, type=str, 
        help="the model type")

    # predication_window
    parser.add_argument("--predication_window", required=True, type=str, 
        help="the predication window, 0 year, 1 year, 3 years, 5 years")

    # number of jobs
    parser.add_argument("--number_of_jobs", required=True, type=str, 
        help="the nb")

    # n_iterations
    parser.add_argument("--n_iterations", required=True, type=str, 
        help="the number of iterations")

    # output_path
    parser.add_argument("--output_path", required=True, type=str, 
        help="the output_path that will be used to save output DDI data")

    args = parser.parse_args()
    main(args)