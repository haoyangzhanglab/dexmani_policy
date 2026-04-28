#!/bin/bash

# 单卡训练示例

# DP3
python dexmani_policy/train.py --config-name=dp3

# MoE-DP3
python dexmani_policy/train.py --config-name=moe_dp3

# DP
python dexmani_policy/train.py --config-name=dp

# ManiFlow单卡
python dexmani_policy/train.py --config-name=maniflow

# ManiFlow多卡（4卡）
python dexmani_policy/train_ddp.py --config-name=maniflow_ddp

# 指定使用特定GPU（例如GPU 1,2,3,4）
python dexmani_policy/train_ddp.py --config-name=maniflow_ddp \
  training.gpu_ids=[1,2,3,4]

# 使用非连续GPU（例如GPU 0和7）
python dexmani_policy/train_ddp.py --config-name=maniflow_ddp \
  training.num_gpus=2 training.gpu_ids=[0,7]

# 使用环境变量指定GPU
CUDA_VISIBLE_DEVICES=2,3,5,6 python dexmani_policy/train_ddp.py \
  --config-name=maniflow_ddp training.num_gpus=4 training.gpu_ids=null

