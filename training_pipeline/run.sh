#!/bin/bash

# training


python training.py \
  --output_path '/Users/cdang/Documents/Github/BMI/colorectal-cancer-predication/' \
  --case_control_path '/Users/cdang/Documents/Github/BMI/colorectal-cancer-predication/crc_data/psm_result/' \
  --case_control_filename 'matched_case_control_CC_01yr' \
  --encoding_path '/Users/cdang/Documents/Github/BMI/colorectal-cancer-predication/crc_data/encoding_files/' \
  --encoding_filename 'data_CC0yr_expr_features' \
  --data_filename 'data_CC0yr_expr' \
  --model_type 'lr' \
  --predication_window 1 \
  --number_of_jobs 10 \
  --n_iterations 20

