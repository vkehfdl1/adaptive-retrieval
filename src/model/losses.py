import itertools
from typing import List

import torch
from autorag.nodes.retrieval.hybrid_cc import hybrid_cc
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
		semantic_ids: List[List[str]],
		lexical_ids: List[List[str]],
		semantic_scores: List[List[float]],
		lexical_scores: List[List[float]],
		predicted_weight: float,
		retrieval_gt_list,
	):
		cc_result_ids, cc_result_scores = hybrid_cc(
			ids=(semantic_ids, lexical_ids),
			scores=(semantic_scores, lexical_scores),
			top_k=self.top_k,
			weight=predicted_weight,
			normalize_method="tmm",
		)

		# Get positive sample scores and negative sample scores
		retrieval_gt_list = [
			list(itertools.chain.from_iterable(retrieval_gt))
			for retrieval_gt in retrieval_gt_list
		]

		positive_score_list, negative_score_list = [], []
		for cc_result_id_list, cc_result_scores_list, retrieval_gt in zip(
			cc_result_ids, cc_result_scores, retrieval_gt_list
		):
			positive_scores = []
			negative_scores = []
			for j, idx in enumerate(cc_result_id_list):
				if idx in retrieval_gt:
					positive_scores.append(cc_result_scores_list[j])
				else:
					negative_scores.append(cc_result_scores_list[j])

			positive_score_list.append(positive_scores)
			negative_score_list.append(negative_scores)

		result_list = []
		for positive_scores, negative_scores in zip(
			positive_score_list, negative_score_list
		):
			positive_scores_tensor = torch.tensor(positive_scores)
			negative_scores_tensor = torch.tensor(negative_scores)
			sum_positive_samples = torch.sum(
				torch.exp(positive_scores_tensor / self.temperature)
			)
			sum_negative_samples = torch.sum(
				torch.exp(negative_scores_tensor / self.temperature)
			)
			loss = -torch.log(
				sum_positive_samples
				/ (sum_positive_samples + sum_negative_samples + 1e-10)
			)
			result_list.append(loss)
		return torch.mean(torch.stack(result_list))
