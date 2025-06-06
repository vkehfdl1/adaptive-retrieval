python3 ./src/reranker/run.py \
--qa_embedding_df_test_path ./data/ko-strategyqa/qa_dev_embeddings.parquet \
--semantic_retrieval_df_test_path ./projects/ko-strategyqa-dev/2/retrieve_node_line/retrieval/0.parquet \
--lexical_retrieval_df_test_path ./projects/ko-strategyqa-dev/2/retrieve_node_line/retrieval/3.parquet \
--save_path ./reranker_zero_or_one_ko-strategyqa-dev_result.parquet
