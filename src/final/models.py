import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MLPRegression(pl.LightningModule):
	def __init__(
		self,
		input_dim,
		embedding_dim,
		latent_dim: int = 128,
		use_score_distribution: bool = False,
		distribution_k: int = 0,
		output_dim=1,
	):
		super().__init__()
		self.save_hyperparameters()

		if use_score_distribution:
			assert distribution_k > 0

		self.use_score_distribution = use_score_distribution
		self.distribution_k = distribution_k

		layers = []
		prev_dim = input_dim

		for hidden_dim in (embedding_dim + distribution_k + distribution_k, latent_dim):
			layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
			prev_dim = hidden_dim

		layers.append(nn.Linear(prev_dim, output_dim))
		self.model = nn.Sequential(*layers)
		self.loss_fn = nn.MSELoss()

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		query_embedding = batch["query_embedding"]  # (B, E)
		semantic_scores = batch["semantic_retrieve_scores"][
			:, : self.distribution_k
		]  # (B, top-k)
		lexical_scores = batch["lexical_retrieve_scores"][
			:, : self.distribution_k
		]  # (B, top-k)

		if self.use_score_distribution:
			input_tensor = torch.cat(
				[query_embedding, semantic_scores, lexical_scores], dim=1
			)
		else:
			input_tensor = query_embedding

		y_hat = self.forward(input_tensor)
		y = batch["gt_weight"]
		loss = self.loss_fn(y_hat, y)
		self.log("train_loss", loss)
		return loss

	def validation_step(self, batch, batch_idx):
		query_embedding = batch["query_embedding"]  # (B, E)
		semantic_scores = batch["semantic_retrieve_scores"][
			:, : self.distribution_k
		]  # (B, top-k)
		lexical_scores = batch["lexical_retrieve_scores"][
			:, : self.distribution_k
		]  # (B, top-k)

		if self.use_score_distribution:
			input_tensor = torch.cat(
				[query_embedding, semantic_scores, lexical_scores], dim=1
			)
		else:
			input_tensor = query_embedding
		y_hat = self(input_tensor)
		y = batch["gt_weight"]
		loss = self.loss_fn(y_hat, y)
		self.log("val_loss", loss)
		return loss

	def test_step(self, batch, batch_idx):
		query_embedding = batch["query_embedding"]  # (B, E)
		semantic_scores = batch["semantic_retrieve_scores"][
			:, : self.distribution_k
		]  # (B, top-k)
		lexical_scores = batch["lexical_retrieve_scores"][
			:, : self.distribution_k
		]  # (B, top-k)

		if self.use_score_distribution:
			input_tensor = torch.cat(
				[query_embedding, semantic_scores, lexical_scores], dim=1
			)
		else:
			input_tensor = query_embedding
		y_hat = self(input_tensor)
		y = batch["gt_weight"]
		loss = self.loss_fn(y_hat, y)
		self.log("test_loss", loss)
		return loss

	def configure_optimizers(self):
		# Initialize AdamW optimizer
		optimizer = optim.AdamW(
			self.parameters(),
			lr=1e-4,  # Initial learning rate
			weight_decay=1e-2,  # Weight decay for regularization
		)

		# Initialize ReduceLROnPlateau scheduler
		scheduler = ReduceLROnPlateau(
			optimizer,
			mode="min",  # 'min' for loss, 'max' for accuracy
			factor=0.1,  # Factor by which to reduce LR
			patience=10,  # Number of epochs to wait before reducing LR
			threshold=0.0001,  # Threshold for measuring improvement
			min_lr=1e-7,  # Minimum learning rate
		)

		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"monitor": "val_loss",  # Metric to monitor
				"frequency": 1,  # How often to check the metric
			},
		}
