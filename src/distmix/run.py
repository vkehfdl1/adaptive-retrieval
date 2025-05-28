import os
from typing import List, Literal

import pandas as pd
import torch
from autorag.nodes.retrieval import BM25
from autorag.utils.util import to_list

from src.data.mfar import AutoRAGQADataset
from src.upper_bound.run import get_non_duplicate_ids, calc_metrics
from src.utils.chroma import ChromaOnlyEmbeddings
from src.utils.hybrid_cc import hybrid_cc, sort_list_to_cc


class DistMixBenchmark:
	def __init__(
		self,
		dataset: AutoRAGQADataset,
		project_dir: str,
		chroma_path: str,
		collection_name: str = "kure",
		top_k: int = 20,
		mix_mode: Literal["sigma", "drop_rr", "weight"] = "sigma",
		mix_sigma_multiplier: float = 1.0,
	):
		self.dataset = dataset
		self.chroma_retrieval = ChromaOnlyEmbeddings(chroma_path, collection_name)
		self.bm25_retrieval = BM25(project_dir, bm25_tokenizer="ko_kiwi")
		self.top_k = top_k
		self.corpus_df = pd.read_parquet(
			os.path.join(project_dir, "data", "corpus.parquet")
		)
		self.mix_mode = mix_mode
		self.mix_sigma_multiplier = mix_sigma_multiplier

	def run(self, save_path: str):
		semantic_ids, lexical_ids, semantic_scores, lexical_scores = (
			self.retrieve_non_duplicate()
		)

		cc_ids, semantic_score_tensor, lexical_score_tensor = sort_list_to_cc(
			semantic_ids, lexical_ids, semantic_scores, lexical_scores
		)  # List[Tensor]
		semantic_scores = [elem.tolist() for elem in semantic_score_tensor]
		lexical_scores = [elem.tolist() for elem in lexical_score_tensor]

		if self.mix_mode == "sigma":
			weights = self.sigma_mix(semantic_scores, lexical_scores)
			weight_tensor = torch.Tensor(weights)
		elif self.mix_mode == "drop_rr":
			weights = self.rr_drop_mix(semantic_scores, lexical_scores)
			weight_tensor = torch.Tensor(weights)
		elif self.mix_mode == "weight":
			raise NotImplementedError
		else:
			raise ValueError(f"Unknown mix mode {self.mix_mode}")

		cc_scores = hybrid_cc(
			weight_tensor, semantic_score_tensor, lexical_score_tensor
		)

		result_cc_ids = []
		result_cc_scores = []
		for cc_id, cc_score_tensor in zip(cc_ids, cc_scores):
			cc_score = cc_score_tensor.tolist()
			temp = [
				(_id, score)
				for _id, score in sorted(
					zip(cc_id, cc_score),
					key=lambda pair: pair[1],
					reverse=True,
				)
			]
			temp_id_result, temp_score_result = zip(*temp)
			result_cc_ids.append(list(temp_id_result)[: self.top_k])
			result_cc_scores.append(list(temp_score_result)[: self.top_k])

		# Now we have cc_ids and cc_scores result with sorted version

		# Calculate Metrics
		metric_df = calc_metrics(
			self.dataset.qa_df["retrieval_gt"].tolist(), result_cc_ids, result_cc_scores
		)

		metric_df["weights"] = weight_tensor.tolist()
		metric_df["retrieved_ids"] = result_cc_ids
		metric_df["retrieved_scores"] = result_cc_scores

		metric_df.to_parquet(save_path, index=False)

	def sigma_mix(self, semantic_retrieve_scores, lexical_retrieve_scores):
		important_semantic_counts = sigma_mix_base(
			semantic_retrieve_scores, self.mix_sigma_multiplier
		)
		important_lexical_counts = sigma_mix_base(
			lexical_retrieve_scores, self.mix_sigma_multiplier
		)

		def decide_weight(semantic_counts, lexical_counts):
			if semantic_counts == 0 and lexical_counts == 0:
				return 0.5

			return (
				(semantic_counts - lexical_counts) / (semantic_counts + lexical_counts)
				+ 1.0
			) / 2.0

		weights = [
			decide_weight(a, b)
			for a, b in zip(important_semantic_counts, important_lexical_counts)
		]
		return weights

	def rr_drop_mix(self, semantic_retrieve_scores, lexical_retrieve_scores):
		semantic_rr = reciprocal_rank_drop(semantic_retrieve_scores)
		lexical_rr = reciprocal_rank_drop(lexical_retrieve_scores)

		def decide_weight(semantic_rr_score, lexical_rr_score):
			return 0.5 + (semantic_rr_score - lexical_rr_score) / 2.0

		weights = [decide_weight(a, b) for a, b in zip(semantic_rr, lexical_rr)]
		return weights

	def retrieve_non_duplicate(self):
		semantic_ids = self.dataset.semantic_df["retrieved_ids"].tolist()
		lexical_ids = self.dataset.lexical_df["retrieved_ids"].tolist()
		semantic_scores = self.dataset.semantic_df["retrieve_scores"].tolist()
		lexical_scores = self.dataset.lexical_df["retrieve_scores"].tolist()
		lexical_target_ids = get_non_duplicate_ids(lexical_ids, semantic_ids)
		semantic_target_ids = get_non_duplicate_ids(semantic_ids, lexical_ids)

		new_semantic_ids, new_semantic_scores = self.chroma_retrieval.get_ids_scores(
			self.dataset.qa_df["query_embeddings"].tolist(), semantic_target_ids
		)
		new_lexical_ids, new_lexical_scores = self.bm25_retrieval._pure(
			self.dataset.qa_df["query"].tolist(), self.top_k, ids=lexical_target_ids
		)

		output_semantic_ids = list(
			map(lambda x, y: to_list(x) + to_list(y), semantic_ids, new_semantic_ids)
		)
		output_lexical_ids = list(
			map(lambda x, y: to_list(x) + to_list(y), lexical_ids, new_lexical_ids)
		)
		output_semantic_scores = list(
			map(
				lambda x, y: to_list(x) + to_list(y),
				semantic_scores,
				new_semantic_scores,
			)
		)
		output_lexical_scores = list(
			map(
				lambda x, y: to_list(x) + to_list(y), lexical_scores, new_lexical_scores
			)
		)

		assert len(output_semantic_ids) == len(output_semantic_scores)
		assert len(output_lexical_ids) == len(output_lexical_scores)

		return (
			output_semantic_ids,  # List[List[]]
			output_lexical_ids,
			output_semantic_scores,
			output_lexical_scores,
		)


def sigma_mix_base(arr: List[List[float]], multiplier: float):
	mean = [sum(row) / len(row) for row in arr]
	std = [
		(sum((x - m) ** 2 for x in row) / len(row)) ** 0.5 for row, m in zip(arr, mean)
	]
	threshold = [m + (s * multiplier) for m, s in zip(mean, std)]

	counts = [sum(1 for x in row if x > t) for row, t in zip(arr, threshold)]
	return counts


def reciprocal_rank_drop(arr: List[List[float]]):
	reciprocal_ranks = []
	for row in arr:
		sorted_row = sorted(row, reverse=True)
		differences = [
			sorted_row[i] - sorted_row[i + 1] for i in range(len(sorted_row) - 1)
		]
		max_difference = max(differences)
		reciprocal_rank = 1 / (differences.index(max_difference) + 1)
		reciprocal_ranks.append(reciprocal_rank)
	return reciprocal_ranks
