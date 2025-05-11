"""
Code from the original paper
https://github.com/microsoft/multifield-adaptive-retrieval/blob/main/mfar/modeling/weighting.py
and modified by me
"""

import torch


class LinearWeights(torch.nn.Module):
	"""
	Linear weights with two options:
	1) Regular mixture of fields
	2) Query conditioning over weights
	"""

	def __init__(self, emb_size):
		super(LinearWeights, self).__init__()
		# emb_size = 1 if no query conditioning
		self.weight = torch.nn.Parameter(torch.ones(emb_size, 1), requires_grad=True)

	def forward(
		self,
		q,  # [Batch, Embedding dimension] # Query Embedding
	) -> torch.Tensor:  # [Batch, Samples]
		"""
		The batch size of x and q must be the same or else this should fail, or the math will be wrong.
		"""
		q = q.float()
		weights = q @ self.weight  # [Batch, Emb] * [Emb, 1] -> [Batch, 1]
		weights_dist = torch.sigmoid(weights)  # [Batch, 1]
		return weights_dist.squeeze()  # [Batch]
