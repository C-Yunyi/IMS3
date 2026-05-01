#!/bin/bash
# run.sh
# 用法: bash run.sh 0.7 1.2   (表示 w_real=0.7, w_sep=1.2)

# 1. 读取命令行参数
W_REAL=$1
W_SEP=$2
FILE_NAME="${W_REAL}x${W_SEP}_test"

# 3. 运行 update_centroid.py
python centroid.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --ckpt pretrained_models/DiT-XL-2-256x256.pt \
  --save-dir ../autodl-tmp/results/dit-distillation/${FILE_NAME} \
  --spec woof \
  --ipc 10 \
  --groups 5 \
  --real-train-dir ../autodl-tmp/imagewoof2/train/ \
  --real-exts .JPEG,.JPG,.PNG .jpeg .jpg .png \
  --w-real ${W_REAL} \
  --w-sep ${W_SEP} \
  --sel-eps 0 \
  --sel-max-iters 5 \
  --feature-backbone resnet18 \
  --seed 0

# 4. 运行 train2.py
python train2.py \
  -d imagenet \
  --imagenet_dir ../autodl-tmp/results/dit-distillation/${FILE_NAME}/final_distilled/train/ ../autodl-tmp/imagewoof2/ \
  -n resnet_ap \
  --nclass 10 \
  --norm_type instance \
  --ipc 10 \
  --tag test \
  --slct_type random \
  --spec woof \
  --randaug true \
  --randaug_n 1 \
  --randaug_m 6
