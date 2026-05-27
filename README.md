# image-edit-flux
## requiement
- python >= 3.12
- cuda >= 13
## Install 
1. Create env
```
python3.12 -m venv .venv
source .venv/bin/activate
```
2. Install packages
```
export HF_TOKEN=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
export VLLM_FLASH_ATTN_VERSION=2

pip install -r requiements.txt
```
3. Start ``vllm`` servecies
### Start from model huggingface
- For GPU RTX PRO 6000
```
.venv/bin/vllm serve black-forest-labs/FLUX.2-klein-9B   --omni   --port 8070
```
- For GPU RTX PRO 5090
```
.venv/bin/vllm serve black-forest-labs/FLUX.2-klein-9B   --omni   --port 8070   --quantization fp8   --gpu-memory-utilization 0.92
```
### Start from local model 
- Download model
```
rclone copy ez_r2_storage:ez-storage/FLUX.2-klein-9B ./models --exclude ".cache/**"  --transfers 8 --checkers 16 --progress --stats 30s
```
- Start service
- For GPU RTX PRO 6000
```
.venv/bin/vllm serve ./models   --omni   --port 8070
```
- For GPU RTX PRO 5090
```
.venv/bin/vllm serve ./models   --omni   --port 8070   --quantization fp8   --gpu-memory-utilization 0.92
```
4. Start api
```
```
VLLM_SERVER_URL=http://localhost:8070 && .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8031
```
