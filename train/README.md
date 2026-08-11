# World-model SFT for BIV (Axolotl branch)

Branch: `Qwen3-Coder-30B-A3B/Axolotl` — QLoRA + **FSDP2** + **context parallel (CP)**.
Sister `*/msswift` uses ms-swift QLoRA + sequence parallel.

## Pipeline

```bash
cd train && source .venv/bin/activate
pip install 'axolotl[ring-flash-attn]' 'bitsandbytes>=0.50'

python scripts/prepare_data.py --all --out-dir data/processed/mix_v2
# Download weights with the matching swift yaml on this branch (configs/swift/).
python scripts/prepare_model.py

export CUDA_VISIBLE_DEVICES=0,1
bash scripts/trainmodel.sh --max-length 8192 --choice 1
```

### Smoke (32k, 2×96GB, Axolotl CP)

```bash
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/trainmodel.sh --max-length 32768 --choice 1
# Expect run yaml: fsdp_version=2, context_parallel_size=2
# wrap: Qwen3MoeDecoderLayer
```

Config: `configs/axolotl/coder_30b_a3b_qlora.yaml`
