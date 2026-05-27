# image-edit-flux

FastAPI wrapper cho FLUX hỗ trợ **tạo ảnh từ văn bản** và **chỉnh sửa ảnh** thông qua REST API.

## Yêu cầu

- Python >= 3.12
- CUDA >= 13

## Cài đặt

**1. Tạo môi trường ảo**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**2. Cài đặt các gói**

```bash
export HF_TOKEN=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
export VLLM_FLASH_ATTN_VERSION=2

pip install -r requiements.txt
```

**3. Khởi động vLLM service**

*Từ model trên Hugging Face*

- GPU RTX PRO 6000:
```bash
.venv/bin/vllm serve black-forest-labs/FLUX.2-klein-9B --omni --port 8070
```

- GPU RTX PRO 5090:
```bash
.venv/bin/vllm serve black-forest-labs/FLUX.2-klein-9B --omni --port 8070 --quantization fp8 --gpu-memory-utilization 0.92
```

*Từ model local*

- Tải model về:
```bash
rclone copy ez_r2_storage:ez-storage/FLUX.2-klein-9B ./models \
  --exclude ".cache/**" --transfers 8 --checkers 16 --progress --stats 30s
```

- Khởi động — GPU RTX PRO 6000:
```bash
.venv/bin/vllm serve ./models --omni --port 8070
```

- Khởi động — GPU RTX PRO 5090:
```bash
.venv/bin/vllm serve ./models --omni --port 8070 --quantization fp8 --gpu-memory-utilization 0.92
```

**4. Khởi động API server**

```bash
VLLM_SERVER_URL=http://localhost:8070 .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8031
```

Hoặc chạy trực tiếp với CLI:

```bash
python server.py --server-url http://localhost:8070 --port 8031
```

### Biến môi trường

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `VLLM_SERVER_URL` | URL của vLLM server | `http://localhost:8070` |
| `REQUEST_TIMEOUT` | Timeout request (giây) | `300` |

---

## API

Base URL: `http://localhost:8031`

Tài liệu Swagger tự động tại: `http://localhost:8031/docs`

---

### `GET /health`

Kiểm tra trạng thái server.

**Response**

```json
{
  "status": "healthy",
  "processing": false,
  "server_url": "http://localhost:8070",
  "timeout": 300
}
```

**Ví dụ**

```bash
curl http://localhost:8031/health
```

---

### `POST /generate`

Tạo hoặc chỉnh sửa ảnh.

- **Không có ảnh đầu vào** → Tạo ảnh từ văn bản (text-to-image)
- **Có ảnh đầu vào** → Chỉnh sửa ảnh (image editing)

Request dạng `multipart/form-data`. Trả về file ảnh JPEG.

**Tham số**

| Tham số | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---------|------|----------|----------|--------|
| `prompt` | string | Có | — | Mô tả ảnh cần tạo/chỉnh sửa |
| `image` | file[] | Không | — | Upload file ảnh (hỗ trợ nhiều ảnh) |
| `image_url` | string[] | Không | — | URL ảnh đầu vào (hỗ trợ nhiều URL) |
| `num_inference_steps` | int | Không | `4` | Số bước inference |
| `guidance_scale` | float | Không | `1.0` | Mức độ bám theo prompt |
| `seed` | int | Không | — | Seed để tái tạo kết quả |
| `negative_prompt` | string | Không | — | Mô tả những gì không muốn có trong ảnh |
| `height` | int | Không | — | Chiều cao ảnh đầu ra (px) |
| `width` | int | Không | — | Chiều rộng ảnh đầu ra (px) |
| `size` | string | Không | — | Kích thước ảnh theo preset (vd: `"1024x1024"`) |
| `num_outputs_per_prompt` | int | Không | — | Số ảnh tạo ra mỗi lần |

---

**Ví dụ 1: Tạo ảnh từ văn bản**

```bash
curl -X POST http://localhost:8031/generate \
  -F "prompt=a beautiful sunset over the ocean, photorealistic" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=3.5" \
  -F "width=1024" \
  -F "height=1024" \
  --output generated.jpg
```

**Ví dụ 2: Chỉnh sửa ảnh bằng file upload**

```bash
curl -X POST http://localhost:8031/generate \
  -F "prompt=change the sky to a stormy night" \
  -F "image=@/path/to/input.jpg" \
  --output edited.jpg
```

**Ví dụ 3: Chỉnh sửa ảnh bằng URL**

```bash
curl -X POST http://localhost:8031/generate \
  -F "prompt=make the person smile" \
  -F "image_url=https://example.com/photo.jpg" \
  --output edited.jpg
```

**Ví dụ 4: Chỉnh sửa ảnh với seed cố định**

```bash
curl -X POST http://localhost:8031/generate \
  -F "prompt=add a hat to the person" \
  -F "image=@/path/to/input.jpg" \
  -F "seed=42" \
  -F "num_inference_steps=20" \
  --output edited.jpg
```

**Ví dụ Python**

```python
import requests

# Tạo ảnh từ văn bản
response = requests.post(
    "http://localhost:8031/generate",
    data={
        "prompt": "a cat sitting on a chair, anime style",
        "num_inference_steps": 20,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024,
    }
)
with open("output.jpg", "wb") as f:
    f.write(response.content)

# Chỉnh sửa ảnh
with open("input.jpg", "rb") as img:
    response = requests.post(
        "http://localhost:8031/generate",
        data={"prompt": "change background to forest"},
        files={"image": img}
    )
with open("output.jpg", "wb") as f:
    f.write(response.content)
```

---

**HTTP Status Codes**

| Code | Mô tả |
|------|--------|
| `200` | Thành công, trả về ảnh JPEG |
| `400` | Thiếu tham số bắt buộc |
| `502` | Lỗi từ vLLM upstream server |
| `500` | Lỗi server nội bộ |
