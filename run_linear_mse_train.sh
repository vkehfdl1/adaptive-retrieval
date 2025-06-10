python3 ./src/linear_mse/train.py --qa_embedding_df_train_path ./data/mldr/qa_train_embeddings.parquet \
--semantic_retrieval_df_train_path ./projects/mldr-train/0/retrieve_node_line/retrieval/0_refined.parquet \
--lexical_retrieval_df_train_path ./projects/mldr-train/0/retrieve_node_line/retrieval/1_refined.parquet \
--upper_bound_df_train_path ./mldr-train_upper_bound_top_k_50.parquet \
--checkpoint_path ./mldr_checkpoints/linear
