python3 ./src/lgbm/train.py --semantic_retrieval_df_path ./projects/mldr-train/0/retrieve_node_line/retrieval/0_refined.parquet \
--lexical_retrieval_df_path ./projects/mldr-train/0/retrieve_node_line/retrieval/1_refined.parquet \
--upper_bound_df_path ./mldr-train_upper_bound_top_k_50.parquet \
--k 20 \
--model_save_path ./mldr_checkpoints/lgbm/lgbm_only_dist_classification.pickle \
--mode classification
