import torch

from src.model.losses import ContrastiveLoss


sample_ids_3 = (
	[
		["id-1", "id-2", "id-3", "id-4", "id-5"],
		["id-2", "id-3", "id-4", "id-5", "id-6"],
	],
	[
		["id-1", "id-4", "id-3", "id-5", "id-2"],
		["id-2", "id-5", "id-4", "id-6", "id-3"],
	],
)
sample_scores_3 = (
	[[5, 3, 1, 0.4, 0.2], [6, 4, 2, 1.4, 1.2]],
	[
		[6, 2, 1, 0.5, 0.1],
		[7, 3, 2, 1.5, 1.1],
	],
)
sample_retrieval_gt_3 = [
	[["id-3"]],
	[["id-2"], ["id-6"]],
]


def test_contrastive_loss_basic():
	"""Test basic functionality of ContrastiveLoss."""
	loss_fn = ContrastiveLoss(sample_limit=6, temperature=0.5)

	predicted_weight = 0.7
	loss = loss_fn(
		sample_ids_3[0],
		sample_ids_3[1],
		sample_scores_3[0],
		sample_scores_3[1],
		predicted_weight,
		sample_retrieval_gt_3,
	)

	assert isinstance(loss, torch.Tensor), "Loss should be a torch Tensor"
	assert loss.item() > 0, "Loss should be positive"
	assert loss.shape == torch.Size([]), "Loss should be a scalar"

	second_loss = loss_fn(
		sample_ids_3[0],
		sample_ids_3[1],
		sample_scores_3[0],
		sample_scores_3[1],
		0.3,
		sample_retrieval_gt_3,
	)
	assert isinstance(second_loss, torch.Tensor), "Loss should be a torch Tensor"
	assert second_loss.item() > 0, "Loss should be positive"
	assert second_loss.shape == torch.Size([]), "Loss should be a scalar"
	assert second_loss != loss, "Loss should be different"
