from typing import Tuple, List

import pandas as pd
import torch


def hybrid_cc(
	weights: torch.Tensor,  # [Batch]
	semantic_scores: List[torch.Tensor],  # [Batch, top-k]
	lexical_scores: List[torch.Tensor],  # [Batch, top-k]
	semantic_theoretical_min_value: float = -1.0,
	lexical_theoretical_min_value: float = 0.0,
) -> List[torch.Tensor]:
	weight_list = weights.tolist()
	result_list = []
	for idx, (semantic_tensor, lexical_tensor) in enumerate(
		zip(semantic_scores, lexical_scores)
	):
		normalize_semantic_scores = normalize_tmm_torch(
			semantic_tensor, semantic_theoretical_min_value
		)
		normalize_lexical_scores = normalize_tmm_torch(
			lexical_tensor, lexical_theoretical_min_value
		)

		weighted_semantic_scores = normalize_semantic_scores * weight_list[idx]
		weighted_lexical_scores = normalize_lexical_scores * (1 - weight_list[idx])

		result = weighted_lexical_scores + weighted_semantic_scores
		result_list.append(result)

	return result_list


def normalize_tmm_torch(scores: torch.Tensor, fixed_min_value: float):
	max_value = torch.max(scores)
	norm_score = (scores - fixed_min_value) / (max_value - fixed_min_value)
	return norm_score


def sort_list_to_cc(
	semantic_ids,
	lexical_ids,
	semantic_scores,
	lexical_scores,
) -> Tuple[List[List[str]], List[torch.Tensor], List[torch.Tensor]]:
	id_result = []
	semantic_score_result = []
	lexical_score_result = []

	for semantic_id, lexical_id, semantic_score, lexical_score in zip(
		semantic_ids, lexical_ids, semantic_scores, lexical_scores
	):
		ids = [semantic_id, lexical_id]
		scores = [semantic_score, lexical_score]
		df = pd.concat(
			[pd.Series(dict(zip(_id, score))) for _id, score in zip(ids, scores)],
			axis=1,
		)
		df.columns = ["semantic", "lexical"]
		df = df.fillna(0)
		df = df.sort_index()
		id_result.append(df.index.tolist())
		semantic_score_result.append(torch.Tensor(df["semantic"].tolist()))
		lexical_score_result.append(torch.Tensor(df["lexical"].tolist()))

	return id_result, semantic_score_result, lexical_score_result
