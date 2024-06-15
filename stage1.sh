#!/bin/bash

# ========================================= rafdb =========================================
CLASS=7
DATASET="rafdb"
DATAPATH="/data/RAFDB/basic/"

python3 train_fer_first_stage.py \
--classes ${CLASS} \
--dataset ${DATASET} \
--data-path ${DATAPATH} \
--alpha 5e-1 \
--lamda 1 \
--topk 16 \
--batch-size 128 \
--gpu 7
