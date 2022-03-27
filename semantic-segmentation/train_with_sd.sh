export NGPUS=4
# train on source data
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/
# train with fine-grained adversarial alignment
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train_adv.py -cfg configs/deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume results/src_r101_try/model_iter020000.pth
# generate pseudo labels for self distillation
# python test.py -cfg configs/deeplabv2_r101_adv.yaml --saveres resume results/adv_test/model_iter040000.pth OUTPUT_DIR datasets/cityscapes/soft_labels DATASETS.TEST cityscapes_train
# train with self distillation
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train_self_distill.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml OUTPUT_DIR results/sd_test