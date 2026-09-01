run_regular() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}
  model_folder=${9}

  echo "Success Rate: regular ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=8 finetune8bh200.py \
    --train_file ${dataset}_train.jsonl --eval_file ${dataset}_test.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} --regular \
    --model_name=${base_model_name} --learning_rate=${learning_rate}
  python3 evaluate.py --model_name=${model_folder} \
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
  model_folder=${9}

  echo "Success Rate: jepa ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=8 finetune8bh200.py \
    --train_file ${dataset}_train.jsonl --eval_file ${dataset}_test.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate}
  python3 evaluate.py --model_name=${model_folder} \
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
# else
#   last_token=-2
# fi

models=(meta-llama/Llama-3.2-1B-Instruct apple/OpenELM-1_1B-Instruct google/gemma-2-2b-it \
        microsoft/phi-1_5 allenai/OLMo-2-0425-1B-Instruct)
non_it_models=(meta-llama/Llama-3.2-1B apple/OpenELM-1_1B google/gemma-2-2b \
               microsoft/phi-1_5 allenai/OLMo-2-0425-1B)
dataset=(synth turk gsm8k spider)

# model_name=meta-llama/Llama-3.1-8B-Instruct
# learning_rate=2e-5
model_name=allenai/OLMo-2-1124-7B-Instruct
learning_rate=8e-5
dataset=synth
lbd=2.0
for predictors in 0 1 2 3 4
do
  for seed in 82 23 37 84 4
  do
    # run_regular ${model_name} ${learning_rate} 4 -2 ${predictors} ${seed} ${lbd} ${dataset}
    model_folder=./ft-j-olmo2-${learning_rate}-${lbd}-${predictors}-${seed}
    run_jepa ${model_name} ${learning_rate} 4 -1 ${predictors} ${seed} ${lbd} ${dataset} ${model_folder}
  done
done
