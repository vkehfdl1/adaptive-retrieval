import pathlib

import numpy as np
import pandas as pd
import torch

from src.data.mfar import AutoRAGQADataset, AutoRAGDataLoader

root_dir = pathlib.Path(__file__).parent.parent.parent
resource_dir = root_dir / "resources"

semantic_result_df = pd.DataFrame(
	{
		"retrieved_ids": [
			np.array(["tung", "tung1", "tung2"]),
			np.array(["jax", "jax1", "jax2"]),
			np.array(["ha", "ver", "tz"]),
		],
		"retrieve_scores": [
			np.array([0.4, 0.5, 0.6]),
			np.array([0.23, 0.12, 0.5]),
			np.array([0.5, 0.2, 0.4]),
		],
	}
)


def test_autorag_dataset():
	# Check the data loader is working
	df = pd.read_parquet(resource_dir / "qa_sample_embeddings.parquet")
	dataset = AutoRAGQADataset(df, semantic_result_df, semantic_result_df)

	loader = AutoRAGDataLoader(dataset=dataset, batch_size=2)

	for batch in loader:
		assert "query" in batch.keys()
		assert "retrieval_gt" in batch.keys()
		assert "query_embeddings" in batch.keys()
		assert "semantic_retrieved_ids" in batch.keys()
		assert "semantic_retrieve_scores" in batch.keys()
		assert "lexical_retrieved_ids" in batch.keys()
		assert "lexical_retrieve_scores" in batch.keys()
		assert len(batch["query"]) == len(batch["retrieval_gt"])
		assert isinstance(batch["query_embeddings"], torch.Tensor)
		assert isinstance(batch["semantic_retrieved_ids"], list)
		assert isinstance(batch["semantic_retrieve_scores"], torch.Tensor)
		assert batch["query_embeddings"].dim() == 2
		assert batch["semantic_retrieve_scores"].shape[1] == 3
