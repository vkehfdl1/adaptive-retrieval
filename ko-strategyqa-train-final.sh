python3 ./src/final/train.py --qa_embedding_df_train_path ./data/ko-strategyqa/qa_train_embeddings.parquet \
--semantic_retrieval_df_train_path ./projects/ko-strategyqa-train/1/retrieve_node_line/retrieval/0.parquet \
--lexical_retrieval_df_train_path ./projects/ko-strategyqa-train/1/retrieve_node_line/retrieval/1.parquet \
--upper_bound_df_train_path ./ko-strategyqa-train_result_top_k_50.parquet \
--qa_embedding_df_test_path ./data/ko-strategyqa/qa_dev_embeddings.parquet \
--semantic_retrieval_df_test_path ./projects/ko-strategyqa-dev/2/retrieve_node_line/retrieval/0.parquet \
--lexical_retrieval_df_test_path ./projects/ko-strategyqa-dev/2/retrieve_node_line/retrieval/3.parquet \
--upper_bound_df_test_path ./ko-strategyqa-dev_result_top_k_50.parquet \
--checkpoint_path ./checkpoint_without_score_dist/classification \
--mode classification
