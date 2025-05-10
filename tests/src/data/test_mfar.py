import pathlib

import pandas as pd

from src.data.mfar import AutoRAGQADataset, AutoRAGDataLoader

root_dir = pathlib.Path(__file__).parent.parent.parent
resource_dir = root_dir / "resources"


def test_autorag_dataset():
	# Check the data loader is working
	df = pd.read_parquet(resource_dir / "qa_sample.parquet", engine="pyarrow")
	dataset = AutoRAGQADataset(df)
	loader = AutoRAGDataLoader(dataset=dataset, batch_size=2)

	for batch in loader:
		assert "query" in batch.keys()
		assert "retrieval_gt" in batch.keys()
		assert len(batch["query"]) == len(batch["retrieval_gt"])
