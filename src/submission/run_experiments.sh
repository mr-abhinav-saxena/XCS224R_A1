#!/bin/bash

# Remove the data directory
rm -rf data

# Function to run an experiment with specified parameters
run_experiment() {
    local env_name="$1"
    local do_dagger="$2"
    local n_iter="$3"

    # Extract base name without '-v4' suffix for policy file
    local base_name="${env_name%-v4}"
    
    # Determine the exp_name prefix and dagger flag
    local exp_prefix="bc"
    local dagger_flag=""
    if [ "$do_dagger" = "true" ]; then
        exp_prefix="dagger"
        dagger_flag="--do_dagger"
    fi

    # Format the short name for exp_name (e.g., Ant -> ant, Walker2d -> walker)
    local short_name=$(echo "$base_name" | tr '[:upper:]' '[:lower:]')
    if [ "$short_name" = "walker2d" ]; then
        short_name="walker"
    fi
    local exp_name="${exp_prefix}_${short_name}"

    echo "Running: env=$env_name, exp_name=$exp_name, n_iter=$n_iter, do_dagger=$do_dagger"
    
    python run_hw1.py \
        --expert_policy_file "xcs224r/policies/experts/${base_name}.pkl" \
        --env_name "$env_name" \
        --exp_name "$exp_name" \
        --n_iter "$n_iter" \
        $dagger_flag \
        --expert_data "xcs224r/expert_data/expert_data_${env_name}.pkl" \
        --video_log_freq -1
}

echo "=== Behavioral Cloning Experiments ==="
run_experiment "Ant-v4" "false" 1
run_experiment "Walker2d-v4" "false" 1
run_experiment "Hopper-v4" "false" 1
run_experiment "HalfCheetah-v4" "false" 1

echo -e "\n=== Dagger Experiments ==="
run_experiment "Ant-v4" "true" 10
run_experiment "Walker2d-v4" "true" 10
run_experiment "Hopper-v4" "true" 10
run_experiment "HalfCheetah-v4" "true" 10
