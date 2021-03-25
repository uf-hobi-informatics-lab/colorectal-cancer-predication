#!/bin/bash

# training

lr_tuned_parameters = {
    'max_iter': range(100, 4100, 500),
    'tol': [0.0001, 0.001, 0.01, 0.1],
    'C': range(1, 50, 2),
    'solver': ['sag', 'saga'],
    'class_weight': [None, 'balanced']
    }

rf_tuned_parameters = {
     'min_samples_leaf': [12,14,16,18,20],
     'max_depth': [28, 32, 36],
     'n_estimators': range(1000, 1200, 50),
     'min_weight_fraction_leaf': [1e-5],
     'min_samples_split': [48,56,64,72,80],
     #'ccp_alpha': np.arange(0, 0.1, 0.02),
     'criterion': ["entropy"]
}

svm_tuned_parameters = {
    'max_iter': range(100, 4100, 500),
    'tol': [0.0001, 0.001, 0.01, 0.1],
    'C': range(1, 50, 2),
    'class_weight': [None, 'balanced']
    }

lgb_tuned_parameters = {
    'num_leaves': [2,4,8,16,32,64,80,96,128],
    'min_data_in_leaf': [2,8,16,32,64,128,256,512,1024],
    'max_depth': [-1, 8, 16, 32, 64],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'num_iterations': range(100, 2000, 100),
    'n_estimators': range(100, 2000, 100),
    'subsample_for_bin': range(20000, 300000, 20000),
    'min_split_gain': [0., 1e-3, 1e-4, 1e-5],
    'min_child_weight': [1e-3, 1e-4, 1e-5],
    'min_child_samples': [2,4,8,16,32,64,128],
    'subsample': np.arange(0.5, 1, 0.1),
    'colsample_bytree': np.arange(0.5, 1, 0.1),
    'reg_alpha': np.arange(0, 1, 0.1),
    'reg_lambda': np.arange(0, 1, 0.1),
    'tree_learner': ['feature', 'data', 'voting'],
    'max_bin': [128, 256, 512],
    'boosting_type': ['dart', 'gbdt'],
    'feature_fraction': np.arange(0.5, 1, 0.1), 
    'bagging_fraction': np.arange(0.5, 1, 0.1)
    }

python training.py \
  --case_control_path '/data/datasets/chong/2021_CRC/psm_result' \
  --case_control_filename 'matched_case_control_CC_01yr' \
  --encoding_path '/data/datasets/chong/2021_CRC/encoding_files' \
  --encoding_filename 'data_CC0yr_expr_features.pkl' \
  --data_filename 'data_CC0yr_expr' \
  --model_type 'lr' \
  --predication_window [1] \
  --number_of_jobs 10 \
  --n_iterations 20 \
