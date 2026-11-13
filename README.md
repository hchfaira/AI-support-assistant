# Enterprise AI Support Assistant

> LLM Engineering / Applied AI / ML Engineering portfolio project — a production-oriented,
> end-to-end intelligent customer support system. Not "a chatbot wrapping an API" — a full
> retrieval, ranking, fine-tuning, optimization and serving stack.

## 1. What this project is

A system that, given a customer support query, will:

1. Classify intent / sentiment / priority.
2. Retrieve relevant documents from a technical knowledge base using **hybrid search**
   (BM25 + dense vector search).
3. **Rerank** candidates with a cross-encoder.
4. Build a grounded context and generate a response with a **fine-tuned (QLoRA) LLM**.
5. Verify citations against the retrieved sources.
6. Flag tickets that must be escalated to a human agent.

Example output:

```json
{
  "intent": "billing_issue",
  "priority": "high",
  "sentiment": "negative",
  "retrieved_documents": ["billing_policy.pdf", "duplicate_payment_procedure.pdf"],
  "answer": "We are sorry about this duplicate charge...",
  "sources": [{"document": "billing_policy.pdf", "page": 12}],
  "escalation_required": true
}
```

## 2. High-level architecture

```text
                   CLIENT / USER
                        |
                 FastAPI Gateway
                        |
                 Query Processing
                        |
             Intent Classification
                        |
                        v
              HYBRID RETRIEVAL
                 /          \
         BM25 Search      Vector Search
                 \          /
                  Candidate Fusion
                        |
                 CROSS-ENCODER RERANKER
                        |
                 Top-K Documents
                        |
                 CONTEXT BUILDER
                        |
                 FINE-TUNED LLM (QLoRA)
                        |
              RESPONSE VALIDATION
              + Citation Verification
                        |
                  JSON RESPONSE
```

Deployment target (see `triton/` and `snowflake/`):

```text
Snowflake (data + model registry + Cortex Search)
        |
Snowpark Container Services
        |
   NVIDIA Triton Inference Server
   ├── Embedding model   (ONNX Runtime backend)
   ├── Reranker model    (ONNX Runtime backend)
   └── Fine-tuned LLM    (vLLM backend)
        |
     FastAPI application
```

## 3. Repository layout

```text
enterprise-ai-support/ (this repo)
├── data/                 # raw / processed / evaluation / sample datasets
├── notebooks/            # model comparison notebooks (see below)
├── src/
│   ├── data_engineering/ # ingestion, parsing, chunking
│   ├── embeddings/       # embedding training/inference/evaluation
│   ├── reranker/         # cross-encoder training/inference/evaluation
│   ├── llm/              # QLoRA fine-tuning, adapter merge, evaluation
│   ├── rag/              # retriever, hybrid search, reranking, generation
│   └── api/              # FastAPI app + Pydantic schemas
├── triton/                # Triton model repository + Dockerfile
├── snowflake/              # SQL + SPCS service spec
├── evaluation/             # retrieval / reranker / RAG / latency metrics
├── monitoring/             # serving metrics + drift detection
└── tests/
```

## 4. Notebooks — model comparison

All notebooks run on CPU with small open models so they are reproducible without a GPU.
Swap the model names for the larger production models (`BAAI/bge-m3`,
`BAAI/bge-reranker-v2-m3`, `Qwen2.5-3B-Instruct`, ...) once running on GPU infrastructure.

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Explore the sample support ticket / knowledge base dataset. |
| `02_embedding_model_comparison.ipynb` | Compare embedding models on Recall@K / MRR@10 / NDCG@10, latency. |
| `03_reranker_comparison.ipynb` | Compare retrieval **with vs without** cross-encoder reranking. |
| `04_llm_finetuning_comparison.ipynb` | Compare base vs QLoRA fine-tuned LLM (ROUGE-L, format compliance). |

## 5. The models

| Component | Reference models | Fine-tuned in this repo? |
|---|---|---|
| Embeddings | `BAAI/bge-m3`, `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2` | Evaluated, optional contrastive fine-tune (`src/embeddings/train.py`) |
| Reranker | `BAAI/bge-reranker-v2-m3` | Fine-tuned with pairwise ranking loss (`src/reranker/train.py`) |
| LLM | `Qwen2.5-3B-Instruct` | Fine-tuned with QLoRA/PEFT (`src/llm/finetuning.py`) |

## 6. Evaluation methodology

Four experiments are compared end-to-end (see `evaluation/`):

| Experiment | Retrieval | Reranker | LLM |
|---|---|---|---|
| Baseline | Vector only | No | Base |
| Experiment 1 | Hybrid (BM25 + Vector) | No | Base |
| Experiment 2 | Hybrid | Yes | Base |
| Experiment 3 | Hybrid | Yes | Fine-tuned |

Metrics tracked:
- **Retrieval**: Recall@5, Recall@10, MRR@10, NDCG@10.
- **RAG**: Faithfulness, Answer Relevancy, Context Precision/Recall, Citation Accuracy (RAGAS/DeepEval).
- **LLM fine-tuning**: Exact Match, ROUGE-L, BERTScore, format compliance.
- **Serving**: throughput, P50/P95/P99 latency, GPU utilization, tokens/sec (`evaluation/latency_benchmark.py`).

## 7. Serving stack

- **NVIDIA Triton Inference Server** hosts three models: ONNX embedding model, ONNX reranker,
  and the fine-tuned LLM via the vLLM backend (`triton/model_repository/`).
- **Snowpark Container Services** packages the Triton image, runs it on a GPU compute pool, and
  exposes a secured endpoint consumed by the FastAPI app (`snowflake/`).
- Snowflake also stores raw/processed documents and can use **Cortex Search** as an alternative,
  managed hybrid-retrieval baseline to compare against the custom pipeline.

## 8. Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the sample hybrid-retrieval + reranking RAG pipeline on the bundled sample KB
python -m src.rag.generation --query "I was charged twice for my Premium subscription"

# Run the API locally
uvicorn src.api.main:app --reload

# Run notebooks
jupyter notebook notebooks/

# Run the test suite
pytest tests/ -v

# Run the full four-experiment evaluation comparison
python -m evaluation.retrieval_metrics
python -m evaluation.reranker_metrics
python -m evaluation.latency_benchmark

# Local full-stack dev (API + Triton, GPU required for Triton)
docker compose up --build
```

All embedding/reranker/LLM-backed modules automatically fall back to lightweight
CPU-only backends (TF-IDF embeddings, lexical-overlap reranking, template-based
generation) when `sentence-transformers`/`torch` or GPU resources aren't
available, so the entire pipeline, API, tests and notebooks above run
end-to-end without a GPU. Swap in the real models (`BAAI/bge-m3`,
`BAAI/bge-reranker-v2-m3`, `Qwen2.5-3B-Instruct`) once running on GPU
infrastructure -- no other code changes required.

## 9. Roadmap (8 weeks)

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Dataset, ingestion, EDA, parsing | Data pipeline |
| 2 | Embeddings, chunking, vector retrieval | First RAG |
| 3 | Hybrid search + reranker fine-tuning | Optimized retrieval |
| 4 | QLoRA LLM fine-tuning | LoRA adapter |
| 5 | Full RAG + evaluation | Working assistant |
| 6 | ONNX, quantization, Triton | Inference server |
| 7 | Docker + Snowpark Container Services | Snowflake deployment |
| 8 | Benchmarking, monitoring, docs | Production-ready project |

## 10. License & data

Verify licensing of any external dataset/model (e.g. Bitext Customer Support dataset,
Banking77) before commercial use or redistribution.
