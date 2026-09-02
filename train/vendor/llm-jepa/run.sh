run_regular() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}

  echo "Success Rate: regular ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=8 finetune.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=./fine-tuned --num_epochs=${epoch} --finetune_seed=${seed} --regular \
    --model_name=${base_model_name} --learning_rate=${learning_rate}
  python evaluate.py --model_name=./fine-tuned \
    --input_file=${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database | tee -a output.txt
}

run_jepa() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}

  echo "Success Rate: jepa ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=8 finetune.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=./fine-tuned --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate}
  python evaluate.py --model_name=./fine-tuned \
    --input_file=${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database | tee -a output.txt
}

# if [[ "$base_model_name" == google/gemma* ]]
# then
#   last_token=-2
# elif [[ "$base_model_name" == apple/OpenELM* ]]
# then
#   last_token=-4
# elif [[ "$base_model_name" == allenai/OLMo-2* ]]
# then
#   last_token=-1
# elif [[ "$base_model_name" == Qwen/Qwen* ]]
# then
#   last_token=-3
# elif [[ "$base_model_name" == deepseek-ai/DeepSeek* ]]
# then
#   last_token=-1
# else
#   last_token=-2
# fi

models=(meta-llama/Llama-3.2-1B-Instruct apple/OpenELM-1_1B-Instruct google/gemma-2-2b-it \
        microsoft/phi-1_5 allenai/OLMo-2-0425-1B-Instruct)
non_it_models=(meta-llama/Llama-3.2-1B apple/OpenELM-1_1B google/gemma-2-2b \
               microsoft/phi-1_5 allenai/OLMo-2-0425-1B)
dataset=(synth turk gsm8k spider)

for seed in 82 23 37 84 4
do
  model_name=meta-llama/Llama-3.2-1B-Instruct
  learning_rate=1e-5
  dataset=gsm8k
  for lbd in 0.5 2.0
  do
    for predictors in 0 1 2
    do
      run_jepa ${model_name} ${learning_rate} 4 -2 ${predictors} ${seed} ${lbd} ${dataset}
    done
  done
done

for seed in 82 23 37 84 4
do
  model_name=meta-llama/Llama-3.2-1B-Instruct
  learning_rate=1e-5
  dataset=spider
  for lbd in 0.5 2.0
  do
    for predictors in 0 1 2
    do
      run_jepa ${model_name} ${learning_rate} 4 -2 ${predictors} ${seed} ${lbd} ${dataset}
    done
  done
done
