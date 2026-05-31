#!/bin/bash

set -e

exec > >(tee logs/pipeline_master.log) 2>&1

echo "======================================"
echo "PIPELINE START"
date
echo "======================================"

python 01_data_audit.py
python 02_data_cleaning.py
python 03_baseline_table.py
python 04_outcome_analysis.py

echo "======================================"
echo "PIPELINE END"
date
echo "======================================"