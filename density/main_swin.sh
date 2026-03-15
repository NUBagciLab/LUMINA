#!/bin/bash

# Default model
model="swin_t"
input_size=224
data_path="/data2/pky0507/dataset/LUMINA_PNG/"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -model|--model)
            model="$2"
            shift 2
            ;;
        -input_size|--input_size)
            input_size="$2"
            shift 2
            ;;
        -data_path|--data_path)
            data_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./main_swin.sh [-model model_name] [-input_size size] [-data_path data_path]"
            exit 1
            ;;
    esac
done

# Loop over folds
for f in {0..4}; do
    python train.py --f "$f" -s 42 --model "$model" --input-size "$input_size" --data-path "$data_path" -j 4 --lr 1e-5
done

python fold_test.py --model "$model" --input-size "$input_size" --data-path "$data_path"