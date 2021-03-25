#!/bin/bash

# training

python data_processing.py \
  --case_control_path '/Users/cdang/Documents/Github/BMI/colorectal-cancer-predication/crc_data/psm_result/' \
  --features_path '/Users/cdang/Documents/Github/BMI/colorectal-cancer-predication/crc_data/encoding_files/' \
  --output_path '/Users/cdang/Documents/Github/BMI/colorectal-cancer-predication/' \
  --case_control_filename 'matched_case_control_CC_01yr' \
  --encoding_filename 'data_CC0yr_expr_features' \
  --data_filename 'data_CC0yr_expr' \
