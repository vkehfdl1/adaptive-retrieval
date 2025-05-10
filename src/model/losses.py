import itertools
from typing import List

import torch
from torch import nn

# 한 batch에 대해서 들어오는 것들
# 이번 step에서 모델의 예측값 (forward 결과) => 하이퍼 파라미터 \alpha
# retrieved ids와 retrieved scores
# top_k (__init__)에서
# retrieval_gt ids => pos - neg sampling을 위함!
# 만약 positive sample이 없는 경우? => loss에 넣기 위해 무조건 positive sample간의 score를 계산하도록 해야함? ㅇㅇ (어짜피 퓨젼)


class ContrastiveLoss(nn.Module):
	def __init__(self, sample_limit: int, temperature: float):
		"""

		:param sample_limit: Same with top_k in Hybrid Retrieval.
			It will be the limitation to samples to use in contrastive loss.
		:param temperature: The temperature hyper-parameter.
		"""
		super().__init__()
		self.top_k = sample_limit
		self.temperature = temperature

	def forward(
		self,
		cc_result_ids: List[List[str]],
		cc_result_scores: List[List[float]],
		retrieval_gt_list,
	):
		# Flatten the retrieval ground truth list
		retrieval_gt_list = [
			list(itertools.chain.from_iterable(retrieval_gt))
			for retrieval_gt in retrieval_gt_list
		]

		# Process the scores
		result_list = []
		for cc_result_id_list, cc_result_scores_list, retrieval_gt in zip(
			cc_result_ids, cc_result_scores, retrieval_gt_list
		):
			# Convert input scores to tensors if they aren't already
			if not isinstance(cc_result_scores_list, torch.Tensor):
				cc_result_scores_tensor = torch.tensor(
					cc_result_scores_list, dtype=torch.float, requires_grad=True
				)
			else:
				cc_result_scores_tensor = cc_result_scores_list

			# Create masks for positive and negative samples
			positive_mask = torch.tensor(
				[idx in retrieval_gt for idx in cc_result_id_list], dtype=torch.bool
			)
			negative_mask = ~positive_mask

			# Apply masks to get positive and negative scores
			positive_scores = cc_result_scores_tensor[positive_mask]
			negative_scores = cc_result_scores_tensor[negative_mask]

			# Skip samples with no positive or negative examples
			if positive_scores.numel() == 0 or negative_scores.numel() == 0:
				continue

			# Calculate loss
			sum_positive_samples = torch.sum(
				torch.exp(positive_scores / self.temperature)
			)
			sum_negative_samples = torch.sum(
				torch.exp(negative_scores / self.temperature)
			)
			loss = -torch.log(
				sum_positive_samples
				/ (sum_positive_samples + sum_negative_samples + 1e-10)
			)
			result_list.append(loss)

		# Return mean loss if we have results, otherwise return zero tensor with grad
		if result_list:
			return torch.mean(torch.stack(result_list))
		else:
			return torch.tensor(0.0, requires_grad=True)
