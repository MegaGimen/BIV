# World-model SFT for BIV (Axolotl branch)

Branch: `Kimi-Dev-72B/Axolotl` — QLoRA + **FSDP2** + **context parallel (CP)**.
Sister `*/msswift` uses ms-swift QLoRA + sequence parallel.

## Pipeline

```bash
cd train && source .venv/bin/activate
pip install 'axolotl[ring-flash-attn]' 'bitsandbytes>=0.50'

python scripts/prepare_data.py --all --out-dir data/processed/mix_v2
python scripts/prepare_model.py

export CUDA_VISIBLE_DEVICES=0,1
bash scripts/trainmodel.sh --max-length 8192 --choice 1
```

### Smoke (32k, 2×96GB, Axolotl CP)

```bash
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/trainmodel.sh --max-length 32768 --choice 1
# Expect run yaml: fsdp_version=2, context_parallel_size=2
# wrap: Qwen2DecoderLayer
```

Config: `configs/axolotl/kimi_dev_72b_qlora.yaml`
