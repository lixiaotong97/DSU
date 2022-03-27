## Multi-domain Generalization


### Prerequisites
+ Create a conda virtual environment and activate it.
```
conda create --name uncertainty python=3.6
conda activate uncertainty
```
+ Install Pytorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
```
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```
+ Install the dependent libraries.
```
pip install flake8 yapf isort yacs gdown tb-nightly future scipy scikit-learn
```
+ Setup the environment
```angular2html
python setup.py develop
```


### Dataset Preparation

Download the datasets PACS ([Download link](https://drive.google.com/file/d/1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE/view)) and Office-Home-DG ([Download link](https://drive.google.com/open?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa)), then place them under the directory like:


```
multi-domain-generalization/DATA
├── pacs
│   ├── images/
│   └── splits
└── office_home_dg
    ├── art/
    ├── clipart/    
    ├── product/    
    └── real_world/
...
```

### Getting Started

We utilize 1 Nvidia Tesla V100 (32G) GPU for training

+ For examples, you can run the following command to train models on `cartoon photo sketch` and test on unseen `art painting`.

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ./DATA \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains cartoon photo sketch \
--target-domains art_painting \
--seed 11 \
--dataset-config-file configs/datasets/dg/pacs.yaml \
--config-file configs/trainers/dg/vanilla/pacs.yaml \
--output-dir output/dg/pacs/uncertainty/art_painting
```
+ You can directly run the full multi-domain generalization experiments on PACS:
```
sh scripts/pacs.sh
```
+ Similarly, you can run the full multi-domain generalization experiments on Office-Home:
```
sh scripts/office_home.sh
```
The checkpoints and logs can be found at [link](https://drive.google.com/drive/folders/1Pn60zo9wAZ656KiuKeqtnkRZNQgbnqcM?usp=sharing).

### Acknowledge

The implementation is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). We thank them for their excellent projects.
