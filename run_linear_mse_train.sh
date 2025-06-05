python3 ./src/linear_mse/train.py --qa_embedding_df_train_path ./data/ko-strategyqa/qa_train_embeddings.parquet \
--semantic_retrieval_df_train_path ./projects/ko-strategyqa-train/1/retrieve_node_line/retrieval/0.parquet \
--lexical_retrieval_df_train_path ./projects/ko-strategyqa-train/1/retrieve_node_line/retrieval/1.parquet \
--upper_bound_df_train_path ./ko-strategyqa-train_result_top_k_50.parquet \
--checkpoint_path ./linear_mse_checkpoint/last.ckpt
