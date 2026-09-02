This directory is a snapshot of https://github.com/galilai-group/llm-jepa
(git history stripped; not a submodule). BIV training does **not** call
`finetune.py`. Stage 1 ports `RepresentationTrainer` wiring (independent
chat-templated Enc(Text)/Enc(Code), `last_token` index, shifted CE +
`1 - cosine`) into `train/scripts/train_jepallm.py`. Pairing is our
`(h,a)` / `o`. `datasets/` and `spider_data.zip` are gitignored.
