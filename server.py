#!/usr/bin/env python3
"""
FastAPI wrapper for FLUX (or any model supporting both text-to-image and image editing)
using the OpenAI-compatible chat API.

This server accepts a prompt with optional input images (file upload or URL),
forwards the request to a vLLM server, and returns the generated/edited image.

- If no image is provided -> text-to-image generation
- If one or more images are provided -> image editing

Environment variables:
    VLLM_SERVER_URL: URL of the vLLM server (default: http://localhost:8070)
    REQUEST_TIMEOUT: Request timeout in seconds (default: 300)

Usage:
    # Via uvicorn
    VLLM_SERVER_URL=http://localhost:8070 uvicorn image_edit_server:app --host 0.0.0.0 --port 8000

    # Direct execution
    python image_edit_server.py --server-url http://localhost:8070 --port 8000
"""

import argparse
import base64
from io import BytesIO
import mimetypes
import os
from pathlib import Path
import threading
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import uvicorn

app = FastAPI(
    title="Image Generation & Edit API",
    description="FastAPI wrapper for FLUX supporting both text-to-image and image editing",
)

# Global state - configurable via environment variables
SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://localhost:8070")
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "300"))

processing = False
processing_lock = threading.Lock()
session = requests.Session()


def _release_processing() -> None:
    global processing
    processing = False
    if processing_lock.locked():
        processing_lock.release()


def _guess_mime_type(image_bytes: bytes, filename: str | None = None) -> str:
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.format:
            return f"image/{img.format.lower()}"
    except Exception:
        pass

    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed

    return "image/png"


def _encode_image_bytes_as_data_url(image_bytes: bytes, filename: str | None = None) -> str:
    mime_type = _guess_mime_type(image_bytes, filename)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def _encode_image_url_as_data_url(image_url: str) -> str:
    if image_url.startswith("data:"):
        return image_url
    if image_url.startswith("http://") or image_url.startswith("https://"):
        response = session.get(image_url, timeout=60)
        response.raise_for_status()
        return _encode_image_bytes_as_data_url(response.content, filename=image_url)

    path = Path(image_url)
    if path.exists():
        return _encode_image_bytes_as_data_url(path.read_bytes(), filename=path.name)

    raise ValueError("Unsupported image_url format")


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    if not data_url.startswith("data:"):
        raise ValueError("Expected data URL response")
    header, b64_data = data_url.split(",", 1)
    mime_type = "application/octet-stream"
    if ";" in header:
        mime_type = header[5:].split(";", 1)[0]
    return base64.b64decode(b64_data), mime_type


def _build_payload(prompt: str, image_data_urls: list[str], extra_body: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for data_url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    payload: dict[str, Any] = {"messages": [{"role": "user", "content": content}]}
    if extra_body:
        payload["extra_body"] = extra_body
    return payload


def _request_image(payload: dict[str, Any]) -> bytes:
    response = session.post(
        f"{SERVER_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list) and content:
        image_url = content[0].get("image_url", {}).get("url", "")
        if image_url.startswith("data:image"):
            image_bytes, _ = _decode_data_url(image_url)
            return image_bytes

    raise ValueError(f"Unexpected response format: {content}")


def _convert_image_bytes_to_jpeg(image_bytes: bytes) -> bytes:
    with Image.open(BytesIO(image_bytes)) as img:
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode == "P":
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            img = background
        else:
            img = img.convert("RGB")

        output = BytesIO()
        img.save(output, format="JPEG", quality=95, optimize=True)
        return output.getvalue()


@app.post("/generate")
def generate_image(
    prompt: str = Form(...),
    image: list[UploadFile] | None = File(None),
    image_url: list[str] | None = Form(None),
    num_inference_steps: int = Form(4),
    guidance_scale: float = Form(1.0),
    seed: int | None = Form(None),
    negative_prompt: str | None = Form(None),
    height: int | None = Form(None),
    width: int | None = Form(None),
    size: str | None = Form(None),
    num_outputs_per_prompt: int | None = Form(None),
) -> StreamingResponse:

    try:
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt parameter")

        # Image input is now optional:
        #   - No image  -> text-to-image generation
        #   - With image -> image editing
        image_data_urls: list[str] = []
        if image:
            for image_file in image:
                # Skip empty file slots (FastAPI may pass empty UploadFile when no file is sent)
                if not image_file or not image_file.filename:
                    continue
                image_bytes = image_file.file.read()
                if not image_bytes:
                    continue
                image_data_urls.append(_encode_image_bytes_as_data_url(image_bytes, image_file.filename))

        if image_url:
            for url in image_url:
                if not url:
                    continue
                image_data_urls.append(_encode_image_url_as_data_url(url))

        extra_body: dict[str, Any] = {}
        if num_inference_steps is not None:
            extra_body["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            extra_body["guidance_scale"] = guidance_scale
        if seed is not None:
            extra_body["seed"] = seed
        if negative_prompt:
            extra_body["negative_prompt"] = negative_prompt
        if height is not None:
            extra_body["height"] = height
        if width is not None:
            extra_body["width"] = width
        if size:
            extra_body["size"] = size
        if num_outputs_per_prompt is not None:
            extra_body["num_outputs_per_prompt"] = num_outputs_per_prompt

        payload = _build_payload(prompt, image_data_urls, extra_body)
        image_bytes = _request_image(payload)
        jpeg_bytes = _convert_image_bytes_to_jpeg(image_bytes)

        return StreamingResponse(
            BytesIO(jpeg_bytes),
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=generated_image.jpeg"},
        )
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Upstream error: {detail}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _release_processing()


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "processing": processing,
        "server_url": SERVER_URL,
        "timeout": REQUEST_TIMEOUT,
    }


def main() -> None:
    """Entry point for direct execution with CLI arguments."""
    global SERVER_URL, REQUEST_TIMEOUT

    parser = argparse.ArgumentParser(description="Image Generation & Edit FastAPI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server (default: 0.0.0.0)")
    parser.add_argument("--server-url", type=str, default=None, help="vLLM server URL (overrides VLLM_SERVER_URL env)")
    parser.add_argument("--timeout", type=int, default=None, help="Request timeout in seconds (overrides REQUEST_TIMEOUT env)")
    args = parser.parse_args()

    # CLI args override environment variables
    if args.server_url:
        SERVER_URL = args.server_url
    if args.timeout:
        REQUEST_TIMEOUT = args.timeout

    print(f"Starting FastAPI server on {args.host}:{args.port}")
    print(f"Upstream vLLM server: {SERVER_URL}")
    print(f"Request timeout: {REQUEST_TIMEOUT}s")
    print("\nAPI Endpoints:")
    print(f"  - POST http://{args.host}:{args.port}/generate")
    print(f"  - GET  http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
