This directory is a snapshot of https://github.com/galilai-group/llm-jepa
(git history stripped; not a submodule). BIV training does **not** call
`finetune.py`. Stage 1 copies the two-loss recipe (token CE + last-token
cosine) into `train/scripts/train_jepallm.py`. `datasets/` and
`spider_data.zip` are gitignored.
