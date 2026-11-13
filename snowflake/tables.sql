-- Tables storing raw/processed documents, chunks, embeddings and model
-- evaluation results for the Enterprise AI Support Assistant.

USE DATABASE ENTERPRISE_AI_SUPPORT;

CREATE TABLE IF NOT EXISTS KNOWLEDGE_BASE.DOCUMENTS (
  doc_id       STRING PRIMARY KEY,
  title        STRING,
  source_path  STRING,
  content_type STRING,  -- 'pdf' | 'html' | 'markdown'
  raw_text     STRING,
  metadata     VARIANT,
  ingested_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS KNOWLEDGE_BASE.CHUNKS (
  chunk_id     STRING PRIMARY KEY,
  doc_id       STRING REFERENCES KNOWLEDGE_BASE.DOCUMENTS(doc_id),
  chunk_text   STRING,
  strategy     STRING,  -- 'fixed-size' | 'overlapping' | 'recursive' | 'semantic'
  metadata     VARIANT,
  created_at   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS KNOWLEDGE_BASE.EMBEDDINGS (
  chunk_id        STRING REFERENCES KNOWLEDGE_BASE.CHUNKS(chunk_id),
  embedding_model STRING,
  embedding       VECTOR(FLOAT, 1024),
  created_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
  PRIMARY KEY (chunk_id, embedding_model)
);

CREATE TABLE IF NOT EXISTS SERVING.TICKETS (
  ticket_id            STRING PRIMARY KEY,
  query                STRING,
  intent               STRING,
  priority             STRING,
  sentiment            STRING,
  retrieved_documents  ARRAY,
  answer               STRING,
  sources              VARIANT,
  escalation_required  BOOLEAN,
  created_at           TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS MODEL_REGISTRY.EVALUATION_RUNS (
  run_id        STRING PRIMARY KEY,
  experiment    STRING,  -- 'baseline' | 'hybrid' | 'hybrid_reranker' | 'hybrid_reranker_finetuned'
  metric_name   STRING,  -- 'recall@5' | 'recall@10' | 'mrr@10' | 'ndcg@10' | 'map' | ...
  metric_value  FLOAT,
  recorded_at   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
