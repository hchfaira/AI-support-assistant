# NVIDIA Triton Inference Server image with the vLLM + ONNX Runtime backends,
# serving the embedding, reranker and fine-tuned LLM models for the
# Enterprise AI Support Assistant.
#
# Build:
#   docker build -t enterprise-ai-support-triton:latest -f triton/Dockerfile .
#
# Run locally (requires an NVIDIA GPU + nvidia-container-toolkit):
#   docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#       -v $(pwd)/triton/model_repository:/models \
#       enterprise-ai-support-triton:latest

FROM nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3

LABEL maintainer="hchfaira"
LABEL description="Triton server for embedding, reranker and fine-tuned LLM models"

WORKDIR /opt/triton

# Extra Python deps used by any Triton Python backend scripts (if added later).
COPY triton/requirements-triton.txt /opt/triton/requirements-triton.txt
RUN pip install --no-cache-dir -r /opt/triton/requirements-triton.txt

# Model repository is mounted at runtime (see docker-compose.yml / SPCS spec)
# so model artifacts don't need to be baked into the image.
ENV MODEL_REPOSITORY=/models

EXPOSE 8000 8001 8002

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:8000/v2/health/ready || exit 1

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models", "--log-verbose=1", "--strict-model-config=false"]
