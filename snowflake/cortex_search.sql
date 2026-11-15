-- Cortex Search service providing a managed, hybrid (lexical + vector +
-- semantic rerank) retrieval baseline, compared against the custom
-- BM25+vector+cross-encoder pipeline in `src/rag/hybrid_search.py`.
--
-- Docs: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview

USE DATABASE ENTERPRISE_AI_SUPPORT;

CREATE OR REPLACE CORTEX SEARCH SERVICE KNOWLEDGE_BASE.SUPPORT_DOCS_SEARCH
  ON chunk_text
  ATTRIBUTES doc_id, strategy
  WAREHOUSE = SUPPORT_ASSISTANT_WH
  TARGET_LAG = '1 hour'
  AS (
    SELECT
      chunk_id,
      chunk_text,
      doc_id,
      strategy
    FROM KNOWLEDGE_BASE.CHUNKS
  );

-- Example query (from the FastAPI service or a Snowpark session):
-- SELECT PARSE_JSON(
--   SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
--     'KNOWLEDGE_BASE.SUPPORT_DOCS_SEARCH',
--     '{"query": "I was charged twice for my subscription", "limit": 10}'
--   )
-- );
