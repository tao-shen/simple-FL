#!/bin/bash

# Experiment Management Script
# This script reads configuration from exp.yml and runs experiments
# Usage: ./exp.sh [config_file]
# Default config file: exp.yml

set -e  # Exit on error

# Configuration file (default: exp.yml)
CONFIG_FILE="${1:-exp.yml}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Load configuration from YAML file
load_config() {
    local config_file="$1"
    python3 << PYTHON_EOF
import yaml
import sys
import json

try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to bash-friendly format
    devices = config.get('devices', [])
    print(f"devices=({' '.join(devices)})")
    
    base_config = config.get('base_config', {})
    print("declare -A base_config")
    for key, value in base_config.items():
        if value is not None:
            print(f"base_config['{key}']='{value}'")
    
    parallel_groups = config.get('parallel_groups', {})
    print("declare -A parallel_experiment_groups")
    for key, values in parallel_groups.items():
        if values:
            print(f"parallel_experiment_groups['{key}']='{' '.join(str(v) for v in values)}'")
    
    serial_groups = config.get('serial_groups', {})
    print("declare -A serial_experiment_groups")
    for key, values in serial_groups.items():
        if values:
            print(f"serial_experiment_groups['{key}']='{' '.join(str(v) for v in values)}'")
    
    device_loop = config.get('device_loop', {})
    print("declare -A device_loop")
    for key, values in device_loop.items():
        if values:
            print(f"device_loop['{key}']='{' '.join(str(v) for v in values)}'")
    
    output = config.get('output', {})
    base_dir = output.get('base_dir', '~/Experiments')
    print(f"output_base_dir='{base_dir}'")
    
except Exception as e:
    print(f"Error loading config: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
}

# Load configuration
eval "$(load_config "$CONFIG_FILE")"

# Function to initialize the experiment configuration
init_exp() {
    declare -A config
    # Copy base config
    for key in "${!base_config[@]}"; do
        config["$key"]="${base_config[$key]}"
    done
    declare -p config
}

# ANSI escape sequences for text color
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'  # No color

# Function to start the experiment
start_exp() {
    local time=$(date "+%Y%m%d-%H%M%S")
    local name=""
    for var in "${!config[@]}"; do
        name+="_${var}_${config[$var]}"
    done
    name=${name:1}  # remove leading underscore
    
    # Get theme from config
    local theme="${config[theme]:-experiments}"
    
    print_config
    mkdir -p "${output_base_dir}/$theme/$name"
    cd "$(dirname "$0")"  # Change to script directory
    PYTHONPATH="$(pwd):${PYTHONPATH}" python3 scripts/train_fl.py \
        $(for var in "${!config[@]}"; do echo "--$var=${config[$var]}"; done) \
        >"${output_base_dir}/$theme/$name/$time.log" \
        2>&1 &
    echo -e "${GREEN}Experiment started!${NC}"
    echo "-------------------------------"
    experiment_counter=$((experiment_counter+1))
}

# Function to print the current configuration
print_config() {
    echo -e "${GREEN}Experiment ${experiment_counter}:${NC}"
    for var in "${!config[@]}"; do
        if [[ ${parallel_experiment_groups[$var]+_} ]]; then
            echo -e "${BLUE}$var: ${config[$var]}${NC} (parallel)"
        elif [[ ${serial_experiment_groups[$var]+_} ]]; then
            echo -e "${YELLOW}$var: ${config[$var]}${NC} (serial)"
        elif [[ ${device_loop[$var]+_} ]]; then
            echo -e "${YELLOW}$var: ${config[$var]}${NC} (device loop)"
        elif [ "$var" == 'device' ]; then
            echo -e "${RED}$var: ${config[$var]}${NC}"
        else
            echo "$var: ${config[$var]}"
        fi
    done
}

# Function to run parallel experiments
run_parallel_experiments() {
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
run_serial_experiments() {
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
            wait  # Wait for all parallel experiments to finish
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
    if [ ${#values[@]} -gt ${#devices[@]} ]; then
        echo "Error: Number of device loop values (${#values[@]}) exceeds number of devices (${#devices[@]})"
        exit 1
    fi
    
    for index in ${!values[@]}; do
        device=${devices[$index]}
        eval "$(init_exp)"
        config['device']="$device"  # Set device
        config[$device_loop_key]="${values[$index]}"  # Set device loop parameter
        run_serial_experiments 0 &
    done
}

# Main script starts here
echo "=========================================="
echo "Experiment Management Script"
echo "Configuration file: $CONFIG_FILE"
echo "=========================================="
echo ""

# Initialize experiment counter
experiment_counter=1

# Start experiments with device loop
run_device_loop

# Wait for all background processes to finish
# Uncomment the line below if you want to wait for all experiments
# wait

echo ""
echo "All experiments have been started!"
echo "Check logs in: ${output_base_dir}"
