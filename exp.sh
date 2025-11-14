#!/bin/bash

# Device list
declare -a devices=("cuda:0" "cuda:1" "cuda:2" "cuda:3")

# Function to initialize the experiment
init_exp() {
    declare -A config
    config=(

        ['theme']='icml25_baselines'
        ['seed']='192'
        ['exp_name']='main_baselines'
        ['device']='cuda:3'
        ['dataset']='femnist'
        ['method']='fedlow'
        ['iid']='natural'
        # ['clients_per_round']=10
        # ['local_epochs']=5
        # ['test_client_ratio']=0.1
    )
    declare -p config
}

# Declare the device loop, parallel and serial experiment groups
declare -A parallel_experiment_groups
parallel_experiment_groups=(
    # ['lora_rank']='4 8 16 32'
    # ['method']='fedavg fedavgm fedprox scaffold fedopt'
    # ['method']='fedcsd elastic'
    # ['seed']='192 225 226 230 231'

    ['method']='fedavg fedavgm fedprox scaffold fedopt'

    # ['method']='fedavg fedavgm fedprox scaffold fedopt fedeve'
    # ['method']='feddyn fedspeed'
    # ['iid']='alpha_1 alpha_0.1 alpha_0.01'
)

declare -A serial_experiment_groups
serial_experiment_groups=(
    ['seed']='192 225 226 230 231'
    # ['method']='fedavg fedavgm fedprox scaffold fedopt'
    # ['dataset']='fashionmnist cifar10 cifar100'
    ['lr_l']='0.001 0.01'
    # ['local_epochs']='1 3 5'
    # ['mu']='0.1 0.01 0.001'
    # ['dataset']='cifar10 cifar100'
    # ['clients_per_round']='10 20 30'
    ['cnn_version']='2layer 3layer'
)

declare -A device_loop
device_loop=(
    ['iid']='natural alpha_1 alpha_0.1 alpha_0.01'
    # ['iid']='alpha_1 alpha_0.1 alpha_0.01'
    # ['seed']='232'
    # ['local_epochs']='1 3 5 10'
)

# ANSI escape sequences for text color
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'  # No color

# Function to start the experiment
start_exp()
{
    local time=$(date "+%Y%m%d-%H%M%S")
    local name=""
    for var in "${!config[@]}"; do
        name+="_${var}_${config[$var]}"
    done
    name=${name:1}  # remove leading underscore
    print_config
    mkdir -p ~/Experiments/$theme/$name 
    python fl_training.py\
        $(for var in "${!config[@]}"; do echo "--$var=${config[$var]}"; done)\
        >~/Experiments/$theme/$name/$time.log \
        2>&1 &
    echo -e "${GREEN}exp_started!${NC}"
    echo "-------------------------------"
    experiment_counter=$((experiment_counter+1))
}

# Function to print the current configuration
print_config()
{
    echo -e "${GREEN}Experiment ${experiment_counter}:${NC}"
    for var in "${!config[@]}"; do
        if [[ ${parallel_experiment_groups[$var]+_} ]]; then
            echo -e "${BLUE}$var: ${config[$var]}${NC}"
        elif [[ ${serial_experiment_groups[$var]+_} ]]; then
            echo -e "${YELLOW}$var: ${config[$var]}${NC}"
        elif [ $var == 'device' ]; then
            echo -e "${RED}$var: ${config[$var]}${NC}"
        else
            echo "$var: ${config[$var]}"
        fi
    done
}

# Function to run parallel experiments
run_parallel_experiments()
{
    local level=$1
    local experiment_keys=(${!parallel_experiment_groups[@]})
    if [[ ${#experiment_keys[@]} -eq 0 ]]; then
        start_exp
        return
    fi
    local values=(${parallel_experiment_groups[${experiment_keys[$level]}]})
    for i in ${!values[@]}; do
        config[${experiment_keys[$level]}]="${values[$i]}"
        if (( level+1 < ${#experiment_keys[@]} )); then
            run_parallel_experiments $((level+1))
        else
            start_exp
        fi
    done
}

# Function to run serial experiments
run_serial_experiments()
{
    local level=$1
    local experiment_keys=(${!serial_experiment_groups[@]})
    if [[ ${#experiment_keys[@]} -eq 0 ]]; then
        run_parallel_experiments 0
        return
    fi
    local values=(${serial_experiment_groups[${experiment_keys[$level]}]})
    for i in ${!values[@]}; do
        config[${experiment_keys[$level]}]="${values[$i]}"
        if (( level+1 < ${#experiment_keys[@]} )); then
            run_serial_experiments $((level+1))
        else
            run_parallel_experiments 0
            wait
        fi
    done
}

# Function to run device loop
run_device_loop() {
    local device_loop_keys=(${!device_loop[@]})
    if [[ ${#device_loop_keys[@]} -eq 0 ]]; then
        eval "$(init_exp)"
        run_serial_experiments 0 &
        return
    fi
    local device_loop_key=${device_loop_keys[0]}
    local values=(${device_loop[$device_loop_key]})

    # Ensure number of device loop values does not exceed the number of devices
    if [ ${#values[@]} -gt ${#devices[@]} ]
    then
        echo "Number of device loop values must be equal to or less than the number of devices"
        exit 1
    fi

    for index in ${!values[@]}
    do
        device=${devices[$index]}
        eval "$(init_exp)"
        config['device']="$device"  # Modify this line to set device-specific config parameters
        config[$device_loop_key]="${values[$index]}"  # Set device loop parameters
        run_serial_experiments 0 &
        # wait
    done
}

# Main script starts here
# Initialize the experiment and start the loops
experiment_counter=1

# Start experiments with device loop
run_device_loop

# Wait for all background processes to finish
# wait