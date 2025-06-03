python3 ./src/lgbm/test.py --model_checkpoint_path ./lgbm_only_dist_regression.pickle \
--qa_embedding_df_test_path ./data/ko-strategyqa/qa_dev_embeddings.parquet \
--semantic_retrieval_df_test_path ./projects/ko-strategyqa-dev/2/retrieve_node_line/retrieval/0.parquet \
--lexical_retrieval_df_test_path ./projects/ko-strategyqa-dev/2/retrieve_node_line/retrieval/3.parquet \
--save_path ./lgbm_ko-strategyqa-dev_regression_result.parquet \
--mode regression
