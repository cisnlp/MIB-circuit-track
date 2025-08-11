#!/bin/bash

cuda=${1:0}
method=${2:-EAP-IG-inputs}
ablation=${3:-patching}
level=${4:-edge}
eval_split=${5:-validation}

VALID_COMBOS=("interpbench_ioi" "gpt2_ioi" "qwen2.5_ioi" "gemma2_ioi" "llama3_ioi" \
                "qwen2.5_mcqa" "gemma2_mcqa" "llama3_mcqa" \
                "llama3_arithmetic_addition" "llama3_arithmetic_subtraction" \
                "gemma2_arc_easy" "llama3_arc_easy" "llama3_arc_challenge")

for model in {gpt2,qwen2.5}; do
    for task in {ioi,mcqa}; do

        # Check if this combination is valid
        combo="${model}_${task}"
        valid=false
        for valid_combo in "${VALID_COMBOS[@]}"; do
            if [[ "$combo" == "$valid_combo" ]]; then
                valid=true
                break
            fi
        done
        
        # Skip iteration if combination is not valid
        if [[ "$valid" == "false" ]]; then
            continue
        fi
        echo "Discovering circuit for $model on $task"

        if [ "$model" = "llama3" ]; then
            batch_size=1
        elif [ "$task" = "arc_easy" ] || [ "$task" = "arc_challenge" ]; then
            batch_size=1
        elif [ "$model" = "gpt2" ] || [ "$model" = "interpbench" ]; then
            batch_size=20
        else
            batch_size=10
        fi

        if [ "$ablation" = "optimal" ]; then
            optimal_ablation_path_str="--optimal-ablation-path ablations/${model}/${task}_oa.pkl"
        else
            optimal_ablation_path_str=""
        fi

        if [ "$task" = "ioi" ]; then
            num_examples_str="--num-examples 1000"
        elif [ "$task" = "mcqa" ]; then
            num_examples_str=""     # use entire dataset
        else
            num_examples_str="--num-examples 100"
        fi

        CUDA_VISIBLE_DEVICES=$cuda python scripts/run_attribution.py \
            --model $model \
            --tasks $task \
            --batch-size $batch_size \
            --method $method \
            --ablation $ablation \
            --level $level \
            $optimal_ablation_path_str \
            --split train \
            --cache_dir "/nfs/gdata/llms/hf-models" \
            $num_examples_str
    done
done

for model in {gpt2,qwen2.5,}; do
    for task in {ioi,mcqa}; do
        for absolute in {True,False}; do

            # Check if this combination is valid
            combo="${model}_${task}"
            valid=false
            for valid_combo in "${VALID_COMBOS[@]}"; do
                if [[ "$combo" == "$valid_combo" ]]; then
                    valid=true
                    break
                fi
            done
            
            # Skip iteration if combination is not valid
            if [[ "$valid" == "false" ]]; then
                continue
            fi
            echo "Evaluating $model on $task (absolute values: $absolute)"

            if [ "$model" = "llama3" ]; then
                batch_size=1
            elif [ "$task" = "arc" ]; then
                batch_size=1
            elif [ "$model" = "gpt2" ]; then
                batch_size=20
            else
                batch_size=10
            fi

            if [ "$absolute" = "True" ]; then
                absolute_str="--absolute"
            else
                absolute_str=""     # Don't use top absolute values
            fi

            CUDA_VISIBLE_DEVICES=$cuda python scripts/run_evaluation.py \
                --model $model \
                --tasks $task \
                --batch_size $batch_size \
                --method $method \
                --ablation $ablation \
                --level $level \
                --split $eval_split \
                --cache_dir "/nfs/gdata/llms/hf-models" \
                $absolute_str 
        done
    done
done