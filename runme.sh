#!/bin/bash
# You need to modify this path
DATASET_DIR="/home/renzhao/dcase2018"

# You need to modify this path as your workspace
WORKSPACE="/home/renzhao/dcase2018/pub_dcase2018_crnn"

DEV_SUBTASK_A_DIR="development-subtaskA"
DEV_SUBTASK_B_DIR="development-subtaskB-mobile"
LB_SUBTASK_A_DIR="leaderboard-subtaskA"
LB_SUBTASK_B_DIR="leaderboard-subtaskB-mobile"
EVAL_SUBTASK_A_DIR="evaluation-subtaskA"
EVAL_SUBTASK_B_DIR="evaluation-subtaskB-mobile"

BACKEND="pytorch"
HOLDOUT_FOLD=1
GPU_ID=0

############ Extract features ############
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_A_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_B_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_A_DIR --data_type=evaluation --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_B_DIR --data_type=evaluation --workspace=$WORKSPACE

############ Development subtask A ############
# Train model for subtask A
python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD --cuda

# Evaluate subtask A
#python $BACKEND/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=15000 --cuda

############ Development subtask B ############
# Train model for subtask B
#python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD --cuda

# Evaluate subtask B
#python $BACKEND/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=15000 --cuda


############ Full train subtask A ############
# Train on full development data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --cuda

# Inference leaderboard data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py inference_leaderboard_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_A_DIR --leaderboard_subdir=$LB_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration=3000 --cuda

# Inference evaluation data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py inference_evaluation_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_A_DIR --eval_subdir=$EVAL_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration=3000 --cuda

############ Full train subtask B ############
# Trian on full development data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --cuda

# Inference leaderboard data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py inference_leaderboard_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_B_DIR --leaderboard_subdir=$LB_SUBTASK_B_DIR --workspace=$WORKSPACE --iteration=3000 --cuda

# Inference evaluation data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py inference_evaluation_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_B_DIR --eval_subdir=$EVAL_SUBTASK_B_DIR --workspace=$WORKSPACE --iteration=3000 --cuda
