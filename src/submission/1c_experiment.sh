#!/bin/bash

# Loop through different values for your hyperparameter
for steps in 100 500 750 1000 1500 2000 5000 7500 10000; do
    echo "------------------------------------------------"
    echo "Running experiment with steps: ${steps}"
    echo "------------------------------------------------"
    
    python run_hw1.py \
        --expert_policy_file xcs224r/policies/experts/Ant.pkl \
        --env_name Ant-v4 \
        --exp_name bc_ant_steps_${steps} \
        --n_iter 1 \
        --expert_data xcs224r/expert_data/expert_data_Ant-v4.pkl \
        --batch_size 10000 \
        --ep_len 1000 \
        --eval_batch_size 10000 \
        --video_log_freq -1 \
        --num_agent_train_steps_per_iter ${steps}
done