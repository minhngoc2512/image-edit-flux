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
- For GPU RTX PRO 6000
```
.venv/bin/vllm serve black-forest-labs/FLUX.2-klein-9B   --omni   --port 8070
```
- For GPU RTX PRO 5090
```
.venv/bin/vllm serve black-forest-labs/FLUX.2-klein-9B   --omni   --port 8070   --quantization fp8   --gpu-memory-utilization 0.92
```
4. Start api
```
VLLM_SERVER_URL=http://localhost:8070 && .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8031
```
