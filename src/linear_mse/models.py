import torch


class LinearWeightsAndBias(torch.nn.Module):
	def __init__(self, emb_size):
		super().__init__()
		# emb_size = 1 if no query conditioning
		self.weight = torch.nn.Parameter(torch.ones(emb_size, 1), requires_grad=True)
		self.bias = torch.nn.Parameter(
			torch.zeros(1), requires_grad=True
		)  # Trainable bias

	def forward(
		self,
		q,  # [Batch, Embedding dimension] # Query Embedding
	) -> torch.Tensor:  # [Batch, Samples]
		"""
		The batch size of x and q must be the same or else this should fail, or the math will be wrong.
		"""
		weights = q @ self.weight  # [Batch, Emb] * [Emb, 1] -> [Batch, 1]
		weights = weights.squeeze()
		weights = weights + self.bias  # Add bias term
		weights_dist = torch.sigmoid(weights)  # [Batch]
		return weights_dist  # [Batch]
