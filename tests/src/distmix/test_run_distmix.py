from pathlib import Path

import pandas as pd
import pytest

from src.data.mfar import AutoRAGQADataset
from src.distmix.run import (
	sigma_mix_base,
	DistMixBenchmark,
	reciprocal_rank_drop,
	weight_method,
)

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


@pytest.fixture
def mldr_dev_dataset():
	project_name = "mldr-dev"
	qa_df = pd.read_parquet(data_dir / "mldr" / "qa_dev_embeddings.parquet")
	semantic_retrieval_df = pd.read_parquet(
		project_dir
		/ project_name
		/ "0"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "0.parquet"
	)
	lexical_retrieval_df = pd.read_parquet(
		project_dir
		/ project_name
		/ "0"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "1.parquet"
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


def test_reciprocal_rank_drop_multiple_rows():
	# Test with multiple rows
	input_data = [
		[10.0, 5.0, 4.0, 3.0],  # max diff between first and second (5.0)
		[7.0, 6.0, 2.0, 1.0],  # max diff between second and third (4.0)
		[8.0, 7.0, 6.0, 1.0],  # max diff between third and fourth (5.0)
	]
	expected = [1.0, 0.5, 1 / 3]
	assert reciprocal_rank_drop(input_data) == expected


def test_weight_method_multiple_rows():
	input_data = [
		[10.0, 5.0, 4.0, 3.0, 2.0],
		[7.0, 6.0, 2.0, 1.0],
		[8.0, 7.0, 6.0, 1.0, 0.4, 0.2],
	]
	expected = [15 + 2 + 1, 3 + 8 + 1, 3 + 2 + 5]
	assert weight_method(input_data, 4) == expected


def test_sigma_mix_run(mldr_dev_dataset):
	dataset = mldr_dev_dataset
	project_name = "mldr-dev"
	top_k = 50
	sigma_multiplier = 2.0

	distmix = DistMixBenchmark(
		dataset,
		project_dir=str(project_dir / project_name),
		chroma_path=str(project_dir / project_name / "resources" / "chroma"),
		top_k=top_k,
		mix_sigma_multiplier=sigma_multiplier,
	)
	distmix.run(
		str(
			root_dir
			/ f"{project_name}_distmix_sigma_{sigma_multiplier}_top_k_{top_k}.parquet"
		),
	)


def test_drop_rr_run(mldr_dev_dataset):
	dataset = mldr_dev_dataset
	project_name = "mldr-dev"
	top_k = 50

	distmix = DistMixBenchmark(
		dataset,
		project_dir=str(project_dir / project_name),
		chroma_path=str(project_dir / project_name / "resources" / "chroma"),
		top_k=top_k,
		mix_mode="drop_rr",
	)
	distmix.run(
		str(root_dir / f"{project_name}_distmix_drop_rr_top_k_{top_k}.parquet"),
	)


def test_weight_mix_run(mldr_dev_dataset):
	dataset = mldr_dev_dataset
	project_name = "mldr-dev"
	top_k = 50

	distmix = DistMixBenchmark(
		dataset,
		project_dir=str(project_dir / project_name),
		chroma_path=str(project_dir / project_name / "resources" / "chroma"),
		top_k=top_k,
		mix_mode="weight",
	)
	distmix.run(
		str(root_dir / f"{project_name}_distmix_weight_top_k_{top_k}.parquet"),
	)
