-- GPU compute pool hosting the Triton inference service via Snowpark
-- Container Services (SPCS). Verify GPU instance family availability in your
-- account's region before running this.
-- Docs: https://docs.snowflake.com/en/developer-guide/snowpark-container-services/working-with-compute-pool

CREATE COMPUTE POOL IF NOT EXISTS SUPPORT_ASSISTANT_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 2
  INSTANCE_FAMILY = 'GPU_NV_S'  -- adjust to an instance family available in your region
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 600;

SHOW COMPUTE POOLS LIKE 'SUPPORT_ASSISTANT_GPU_POOL';
DESCRIBE COMPUTE POOL SUPPORT_ASSISTANT_GPU_POOL;
