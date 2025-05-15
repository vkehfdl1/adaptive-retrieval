# This is a python code for finding empirical upper bound of query adaptive hybrid retrieval
from typing import List

import pandas as pd
import torch
from autorag.evaluation import evaluate_retrieval
from autorag.nodes.retrieval import BM25
from autorag.schema.metricinput import MetricInput
from tqdm import tqdm

from src.data.mfar import AutoRAGQADataset
from src.model.losses import ContrastiveLoss
from src.utils.chroma import ChromaOnlyEmbeddings
from src.utils.hybrid_cc import sort_list_to_cc, hybrid_cc


class UpperBoundFinder:
	def __init__(
		self,
		dataset: AutoRAGQADataset,
		project_dir: str,
		chroma_path: str,
		collection_name: str = "kure",
		top_k: int = 20,
	):
		self.dataset = dataset
		self.chroma_retrieval = ChromaOnlyEmbeddings(chroma_path, collection_name)
		self.bm25_retrieval = BM25(project_dir, bm25_tokenizer="ko_kiwi")
		self.top_k = top_k
		self.loss = ContrastiveLoss(
			sample_limit=self.top_k,
			temperature=1.0,
		)

	def run(self, save_path: str, save_at_every_step: int = 1):
		if not save_path.endswith(".parquet"):
			raise ValueError("save_path must end with .parquet")

		result_dict_list = []
		for idx in tqdm(range(len(self.dataset))):
			batch = self.dataset[idx]
			retrieval_gt = batch["retrieval_gt"]
			semantic_ids, lexical_ids, semantic_scores, lexical_scores = (
				self.retrieve_non_duplicate(batch)
			)
			cc_ids, semantic_score_tensor, lexical_score_tensor = sort_list_to_cc(
				semantic_ids, lexical_ids, semantic_scores, lexical_scores
			)

			weight_tensor = torch.linspace(0.0, 1.00, steps=101)
			cc_scores = hybrid_cc(
				weight_tensor,
				[semantic_score_tensor[0] for _ in range(101)],
				[lexical_score_tensor[0] for _ in range(101)],
			)
			contrastive_loss_list = [
				self.loss(cc_ids, [cc_score.tolist()], [retrieval_gt])
				for cc_score in cc_scores
			]
			best_weight_idx = torch.argmin(
				torch.tensor([t.item() for t in contrastive_loss_list])
			).item()

			# Evaluate the best weight
			metric_df = calc_metrics(
				[retrieval_gt], cc_ids, [cc_scores[best_weight_idx].tolist()]
			)

			metric_dict = {
				"f1": metric_df["retrieval_f1"].mean(),
				"precision": metric_df["retrieval_precision"].mean(),
				"recall": metric_df["retrieval_recall"].mean(),
				"ndcg": metric_df["retrieval_ndcg"].mean(),
				"map": metric_df["retrieval_map"].mean(),
				"mrr": metric_df["retrieval_mrr"].mean(),
			}
			metric_dict["loss"] = contrastive_loss_list[best_weight_idx].item()
			metric_dict["best_weight"] = best_weight_idx / 100.0
			metric_dict["query"] = batch["query"]
			metric_dict["retireval_gt"] = retrieval_gt

			result_dict_list.append(metric_dict)
			if idx % save_at_every_step == 0:
				result_df = pd.DataFrame(result_dict_list)
				result_df.to_parquet(save_path, index=False)

	def retrieve_non_duplicate(self, batch):
		semantic_ids = [batch["semantic_retrieved_ids"]]
		lexical_ids = [batch["lexical_retrieved_ids"]]
		semantic_scores = [batch["semantic_retrieve_scores"].tolist()]
		lexical_scores = [batch["lexical_retrieve_scores"].tolist()]
		lexical_target_ids = get_non_duplicate_ids(lexical_ids, semantic_ids)
		semantic_target_ids = get_non_duplicate_ids(semantic_ids, lexical_ids)

		new_semantic_ids, new_semantic_scores = self.chroma_retrieval.get_ids_scores(
			[batch["query_embeddings"].tolist()], semantic_target_ids
		)
		new_lexical_ids, new_lexical_scores = self.bm25_retrieval._pure(
			[batch["query"]], self.top_k, ids=lexical_target_ids
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
		assert len(output_semantic_ids) == 1

		return (
			output_semantic_ids,  # List[List[]]
			output_lexical_ids,
			output_semantic_scores,
			output_lexical_scores,
		)


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


def calc_metrics(retrieval_gt, id_result, score_result):
	@evaluate_retrieval(
		metric_inputs=list(
			map(
				lambda x: MetricInput(retrieval_gt=x),
				retrieval_gt,
			)
		),
		metrics=[
			"retrieval_f1",
			"retrieval_recall",
			"retrieval_precision",
			"retrieval_ndcg",
			"retrieval_map",
			"retrieval_mrr",
		],
	)
	def calculate_metrics():
		return (
			["" for _ in range(len(id_result))],
			id_result,
			score_result,
		)

	return calculate_metrics()
