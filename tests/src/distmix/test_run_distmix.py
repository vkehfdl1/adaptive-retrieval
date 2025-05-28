from pathlib import Path

import pandas as pd
import pytest

from src.data.mfar import AutoRAGQADataset
from src.distmix.run import sigma_mix_base, DistMixBenchmark

root_dir = Path(__file__).parent.parent.parent.parent
data_dir = root_dir / "data"
project_dir = root_dir / "projects"


@pytest.fixture
def allganize_dataset():  # top-k 20
	project_name = "allganize"
	qa_df = pd.read_parquet(data_dir / project_name / "qa_embeddings.parquet")
	semantic_retrieval_df = pd.read_parquet(
		project_dir
		/ project_name
		/ "0"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "1.parquet"
	)
	lexical_retrieval_df = pd.read_parquet(
		project_dir
		/ project_name
		/ "0"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "4.parquet"
	)
	return AutoRAGQADataset(qa_df, semantic_retrieval_df, lexical_retrieval_df)


@pytest.fixture
def ko_strategyqa_dev_dataset():
	project_name = "ko-strategyqa-dev"
	qa_df = pd.read_parquet(data_dir / "ko-strategyqa" / "qa_dev_embeddings.parquet")
	semantic_retrieval_df = pd.read_parquet(
		project_dir
		/ project_name
		/ "2"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "0.parquet"
	)
	lexical_retrieval_df = pd.read_parquet(
		project_dir
		/ project_name
		/ "2"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "3.parquet"
	)
	return AutoRAGQADataset(qa_df, semantic_retrieval_df, lexical_retrieval_df)


def test_sigma_mix_base():
	# Test input
	arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	multiplier = 1.0
	# Actual output
	actual_counts = sigma_mix_base(arr, multiplier)
	expected_counts = [1, 1, 1]
	# Assertion
	assert (
		actual_counts == expected_counts
	), f"Expected {expected_counts}, got {actual_counts}"


def test_sigma_mix_run(allganize_dataset):
	dataset = allganize_dataset
	project_name = "allganize"
	top_k = 20

	distmix = DistMixBenchmark(
		dataset,
		project_dir=str(project_dir / project_name),
		chroma_path=str(project_dir / project_name / "resources" / "chroma"),
		top_k=20,
	)
	distmix.run(
		str(root_dir / f"{project_name}_distmix_sigma_1.0_top_k_{top_k}.parquet"),
	)
