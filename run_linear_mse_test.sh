#!/bin/bash

# Path configurations
MODEL_CHECKPOINT="./linear_mse_checkpoint/last.ckpt"
QA_EMBEDDINGS="./data/ko-strategyqa/qa_dev_embeddings.parquet"
SEMANTIC_RETRIEVAL="./projects/ko-strategyqa-dev/2/retrieve_node_line/retrieval/0.parquet"
LEXICAL_RETRIEVAL="./projects/ko-strategyqa-dev/2/retrieve_node_line/retrieval/3.parquet"
SAVE_PATH="./ko-strategyqa-dev_linear_mse_result.parquet"

# Run the CLI
python -m src.linear_mse.test \
  --model_checkpoint_path "$MODEL_CHECKPOINT" \
  --qa_embedding_df_test_path "$QA_EMBEDDINGS" \
  --semantic_retrieval_df_test_path "$SEMANTIC_RETRIEVAL" \
  --lexical_retrieval_df_test_path "$LEXICAL_RETRIEVAL" \
  --save_path "$SAVE_PATH" \
