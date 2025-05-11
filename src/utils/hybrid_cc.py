import torch


def hybrid_cc(
	weights: torch.Tensor,  # [Batch]
	semantic_scores: torch.Tensor,  # [Batch, top-k]
	lexical_scores: torch.Tensor,  # [Batch, top-k]
	semantic_theoretical_min_value: float = -1.0,
	lexical_theoretical_min_value: float = 0.0,
):
	normalize_semantic_scores = normalize_tmm_torch(
		semantic_scores, semantic_theoretical_min_value
	)
	normalize_lexical_scores = normalize_tmm_torch(
		lexical_scores, lexical_theoretical_min_value
	)

	weighted_semantic_scores = normalize_semantic_scores * weights.view(-1, 1)
	lexical_weights = 1 - weights
	weighted_lexical_scores = normalize_lexical_scores * lexical_weights.view(-1, 1)

	result = weighted_lexical_scores + weighted_semantic_scores
	return result


def normalize_tmm_torch(scores: torch.Tensor, fixed_min_value: float):
	max_value = torch.max(scores)
	norm_score = (scores - fixed_min_value) / (max_value - fixed_min_value)
	return norm_score
