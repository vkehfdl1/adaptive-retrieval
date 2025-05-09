import itertools
from typing import List, Optional

import pytorch_lightning as pl
import torch
from autorag.nodes.retrieval import VectorDB, BM25
from autorag.nodes.retrieval.hybrid_cc import fuse_per_query

from src.model.models import LinearWeights


def get_non_duplicate_ids(target_ids, compare_ids) -> List[List[str]]:
	"""
	Get non-duplicate ids from target_ids and compare_ids.
	If you want to non-duplicate ids of semantic_ids, you have to put it at target_ids.
	"""
	result_ids = []
	assert len(target_ids) == len(compare_ids)
	for target_id_list, compare_id_list in zip(target_ids, compare_ids):
		query_duplicated = list(set(compare_id_list) - set(target_id_list))
		duplicate_list = query_duplicated if len(query_duplicated) != 0 else []
		result_ids.append(duplicate_list)
	return result_ids


class MfarTrainingModule(pl.LightningModule):
	def __init__(
		self,
		project_dir: str,
		vectordb_name: str = "kure",
		top_k: int = 20,
	):
		super().__init__()
		self.vectordb_retrieval = VectorDB(project_dir, vectordb_name)
		self.bm25_retrieval = BM25(project_dir, bm25_tokenizer="ko_kiwi")
		self.embedding_dimension = 1024
		self.top_k = top_k

		self.layer = LinearWeights(self.embedding_dimension)

	def forward(self, x):
		# Input x: [Batch] (string list)
		# Inference
		with torch.no_grad():
			embedding_list = (
				self.vectordb_retrieval.embedding_model._get_text_embeddings(x)
			)
			predicted_weight: torch.Tensor = self.layer(torch.Tensor(embedding_list))

			# Execute Hybrid CC
			semantic_ids, lexical_ids, semantic_scores, lexical_scores = self.retrieve(
				x, retrieval_gt=None
			)
			result_ids, result_scores = [], []
			for idx, weight in enumerate(predicted_weight.tolist()):
				result_id_list, result_score_list = fuse_per_query(
					semantic_ids[idx],
					lexical_ids[idx],
					semantic_scores[idx],
					lexical_scores[idx],
					top_k=self.top_k,
					weight=weight,
					normalize_method="tmm",
					semantic_theoretical_min_value=-1.0,
					lexical_theoretical_min_value=0.0,
				)
				result_ids.append(result_id_list)
				result_scores.append(result_score_list)

			return {
				"ids": result_ids,
				"scores": result_scores,
			}

	def training_step(self, batch, batch_idx):
		# queries = batch["queries"]
		pass

	def retrieve(self, queries: List[str], retrieval_gt: Optional[List[List]] = None):
		"""The result can be `unsorted`."""
		# Several input batch
		input_queries = list(map(lambda x: [x], queries))
		semantic_ids, semantic_scores = self.vectordb_retrieval._pure(
			input_queries, self.top_k
		)
		lexical_ids, lexical_scores = self.bm25_retrieval._pure(
			input_queries, self.top_k
		)

		lexical_target_ids = get_non_duplicate_ids(lexical_ids, semantic_ids)
		semantic_target_ids = get_non_duplicate_ids(semantic_ids, lexical_ids)

		new_semantic_ids, new_semantic_scores = self.vectordb_retrieval._pure(
			input_queries, self.top_k, ids=semantic_target_ids
		)
		new_lexical_ids, new_lexical_scores = self.bm25_retrieval._pure(
			input_queries, self.top_k, ids=lexical_target_ids
		)

		output_semantic_ids = list(
			map(lambda x, y: x + y, semantic_ids, new_semantic_ids)
		)
		output_lexical_ids = list(map(lambda x, y: x + y, lexical_ids, new_lexical_ids))
		output_semantic_scores = list(
			map(lambda x, y: x + y, semantic_scores, new_semantic_scores)
		)
		output_lexical_scores = list(
			map(lambda x, y: x + y, lexical_scores, new_lexical_scores)
		)

		assert len(output_semantic_ids) == len(output_semantic_scores)
		assert len(output_lexical_ids) == len(output_lexical_scores)
		assert len(output_semantic_ids) == len(queries)

		if retrieval_gt is not None:
			retrieval_gt = [
				list(itertools.chain.from_iterable(x)) for x in retrieval_gt
			]
			# Add Retrieval gt when there is no positive sample
			for idx in range(len(output_semantic_ids)):
				if len(set(retrieval_gt[idx]) & set(output_semantic_scores[idx])) < 1:
					positive_id = retrieval_gt[idx][0]
					positive_id_list, positive_score_list = (
						self.vectordb_retrieval._pure(
							[input_queries[idx]], self.top_k, ids=[[positive_id]]
						)
					)
					output_semantic_ids[idx].append(positive_id_list[0][0])
					output_semantic_scores[idx].append(positive_score_list[0][0])

			for idx in range(len(output_lexical_ids)):
				if len(set(retrieval_gt[idx]) & set(output_lexical_scores[idx])) < 1:
					positive_id = retrieval_gt[idx][0]
					positive_id_list, positive_score_list = self.bm25_retrieval._pure(
						[input_queries[idx]], self.top_k, ids=[[positive_id]]
					)
					output_lexical_ids[idx].append(positive_id_list[0][0])
					output_lexical_scores[idx].append(positive_score_list[0][0])

		return (
			output_semantic_ids,  # List[List[]]
			output_lexical_ids,
			output_semantic_scores,
			output_lexical_scores,
		)
