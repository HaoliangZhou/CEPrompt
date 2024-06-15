#!/bin/bash

# ========================================= rafdb =========================================
DATASET='rafdb'
DATAPATH='/data/RAFDB/basic/'
CKPTPATH='/data/RAFDB/ckpt/model_epoch_34_acc9221.pth'

python3 train_fer_second_stage.py \
--dataset ${DATASET} \
--data-path ${DATAPATH} \
--ckpt-path ${CKPTPATH} \
--alpha=5e-1 \
--stage2_name="cat" \
--ctxinit="a photo of" \
--prompts_depth=9 \
--n_ctx=2 \
--gpu=7 \
--epochs=10 \
--warmup_epochs=2 \
--batch-size 128 \
--lr=2e-3