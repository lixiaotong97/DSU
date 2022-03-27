export NGPUS=4
# train on source data
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/deeplabv2_ur101_src.yaml OUTPUT_DIR results/uncertainty