import torch

from src.utils.hybrid_cc import normalize_tmm_torch, hybrid_cc


def test_normalize_tmm_torch_():
	# Test with a simple 1D tensor
	scores = torch.tensor([1.0, 2.0, 3.0])
	fixed_min_value = 0.0
	normalized = normalize_tmm_torch(scores, fixed_min_value)
	expected = torch.Tensor([0.3333, 0.6667, 1.0])
	assert torch.allclose(
		normalized, expected, rtol=1e-3
	), "normalize_tmm_torch did not normalize correctly"

	# Test with a 2D tensor
	scores_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
	normalized_2d = normalize_tmm_torch(scores_2d, fixed_min_value)
	expected_2d = torch.Tensor([[0.25, 0.5], [0.75, 1.0]])
	assert torch.allclose(
		normalized_2d, expected_2d
	), "normalize_tmm_torch failed with 2D input"

	# Test with negative min value
	neg_min = -1.0
	normalized_neg = normalize_tmm_torch(scores, neg_min)
	expected_neg = torch.Tensor([0.5, 0.75, 1.0])
	assert torch.allclose(
		normalized_neg, expected_neg
	), "normalize_tmm_torch failed with negative min value"


def test_hybrid_cc():
	# Test with sample inputs
	weights = torch.tensor([0.3, 0.7])
	semantic_scores = [torch.tensor([2.0, 4.0]), torch.Tensor([1.0, 3.0])]
	# normalize => [[0.6, 1.0], [0.5, 1.0]]
	lexical_scores = [torch.tensor([1.0, 4.0]), torch.Tensor([5.0, 2.0])]
	# normalize => [[0.25, 1.0], [1.0, 0.4]]

	result = hybrid_cc(weights, semantic_scores, lexical_scores)
	expected = [torch.Tensor([0.355, 1.0]), torch.Tensor([0.35 + 0.3, 0.7 + 0.12])]
	assert all(
		torch.allclose(res, exp) for res, exp in zip(result, expected)
	), "hybrid_cc did not compute the expected weighted sum"

	# Test with different weights
	weights2 = torch.tensor([0.9, 0.1])
	result2 = hybrid_cc(weights2, semantic_scores, lexical_scores)
	expected2 = [
		torch.Tensor([0.54 + 0.025, 1.0]),
		torch.Tensor([0.05 + 0.9, 0.1 + 0.36]),
	]

	assert all(
		torch.allclose(res, exp) for res, exp in zip(result2, expected2)
	), "hybrid_cc did not compute the expected weighted sum"
