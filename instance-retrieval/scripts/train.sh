#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
SEED=$4
NAME=$5


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/source_pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} --seed ${SEED} --margin 0.0 \
	--num-instances 4 -b 64 -j 8 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 100 --epochs 80 --eval-step 10 \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-${SEED}-${NAME}
