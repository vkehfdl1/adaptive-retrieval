python3 ./src/lgbm/train.py --semantic_retrieval_df_path ./projects/ko-strategyqa-train/1/retrieve_node_line/retrieval/0.parquet \
--lexical_retrieval_df_path ./projects/ko-strategyqa-train/1/retrieve_node_line/retrieval/1.parquet \
--upper_bound_df_path ./ko-strategyqa-train_result_top_k_50.parquet \
--k 20 \
--model_save_path ./lgbm_only_dist_regression.pickle \
--mode regression
