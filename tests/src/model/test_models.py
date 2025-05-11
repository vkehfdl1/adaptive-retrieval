import itertools
from pathlib import Path

import pandas as pd
import torch

from src.model.train import MfarTrainingModule


root_dir = Path(__file__).parent.parent.parent.parent
project_dir = root_dir / "projects" / "allganize"


def test_training_module_retrieve():
	training_module = MfarTrainingModule(
		str(project_dir), str(project_dir / "resources" / "chroma")
	)
	qa_df = pd.read_parquet(
		root_dir / "data" / "allganize" / "qa_embeddings.parquet", engine="pyarrow"
	)
	input_queries = qa_df["query"].iloc[:5].tolist()
	input_retrieval_gt = qa_df["retrieval_gt"].iloc[:5].tolist()
	input_query_embeddings = qa_df["query_embeddings"].iloc[:5].tolist()

	semantic_ids, lexical_ids, semantic_scores, lexical_scores = (
		training_module.retrieve(
			input_queries, input_query_embeddings, input_retrieval_gt
		)
	)

	assert (
		len(semantic_ids)
		== len(semantic_scores)
		== len(lexical_ids)
		== len(lexical_scores)
		== 5
	)
	assert all(
		len(lst1) == len(lst2) for lst1, lst2 in zip(semantic_scores, semantic_ids)
	)
	assert all(
		len(lst1) == len(lst2) for lst1, lst2 in zip(lexical_scores, lexical_ids)
	)

	for idx, gt_list in enumerate(input_retrieval_gt):
		gt_list = list(itertools.chain.from_iterable(gt_list))
		assert len(set(semantic_ids[idx]) & set(gt_list)) > 0
		assert len(set(lexical_ids[idx]) & set(gt_list)) > 0


def test_training_module_forward():
	training_module = MfarTrainingModule(
		str(project_dir), str(project_dir / "resources" / "chroma")
	)
	sample_queries = [
		"최강 삼성 히어로 누구 김영웅!",
		"중대 기계 재학생 누구 노동건!",
		"동굴형 동굴형! 두산의! 동굴형!",
	]
	sample_query_embeddings = torch.rand((3, 1024))
	result = training_module(
		{"query": sample_queries, "query_embeddings": sample_query_embeddings}
	)
	assert "ids" in result.keys()
	assert "scores" in result.keys()
	assert len(result["ids"]) == len(sample_queries) == len(result["scores"])
	assert len(result["ids"][0]) == 20
	assert len(result["scores"][0]) == 20
