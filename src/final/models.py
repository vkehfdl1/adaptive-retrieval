from typing import Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MLPRegression(pl.LightningModule):
	def __init__(
		self,
		embedding_dim,
		latent_dim: Tuple[int] = (512, 128),
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
		prev_dim = embedding_dim + distribution_k + distribution_k

		for hidden_dim in latent_dim:
			layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
			prev_dim = hidden_dim

		layers.append(nn.Linear(prev_dim, output_dim))
		self.model = nn.Sequential(*layers)
		self.loss_fn = nn.MSELoss()

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		query_embedding = batch["query_embeddings"]  # (B, E)
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
		y_hat = y_hat.squeeze()
		y = batch["gt_weight"]
		loss = self.loss_fn(y_hat, y)
		self.log("train_loss", loss)
		return loss

	def validation_step(self, batch, batch_idx):
		query_embedding = batch["query_embeddings"]  # (B, E)
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
		y_hat = y_hat.squeeze()
		y = batch["gt_weight"]
		loss = self.loss_fn(y_hat, y)
		self.log("val_loss", loss)
		return loss

	def test_step(self, batch, batch_idx):
		query_embedding = batch["query_embeddings"]  # (B, E)
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
		y_hat = y_hat.squeeze()
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


class MLPClassifier(pl.LightningModule):
	def __init__(
		self,
		embedding_dim,
		latent_dim: Tuple[int] = (512, 128),
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
		prev_dim = embedding_dim + distribution_k + distribution_k

		for hidden_dim in latent_dim:
			layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
			prev_dim = hidden_dim

		layers.append(nn.Linear(prev_dim, output_dim))
		layers.append(nn.Sigmoid())
		self.model = nn.Sequential(*layers)
		self.loss_fn = nn.BCELoss()

		# Define validation metrics
		self.val_accuracy = torchmetrics.Accuracy(task="binary")
		self.val_f1 = torchmetrics.F1Score(task="binary")
		self.val_recall = torchmetrics.Recall(task="binary")
		self.val_precision = torchmetrics.Precision(task="binary")

		# Define test metrics (optional)
		self.test_accuracy = torchmetrics.Accuracy(task="binary")
		self.test_f1 = torchmetrics.F1Score(task="binary")
		self.test_recall = torchmetrics.Recall(task="binary")
		self.test_precision = torchmetrics.Precision(task="binary")

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		query_embedding = batch["query_embeddings"]  # (B, E)
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
		y_hat = y_hat.squeeze()
		y = batch["gt_label"]
		loss = self.loss_fn(y_hat, y)
		self.log("train_loss", loss)
		return loss

	def validation_step(self, batch, batch_idx):
		query_embedding = batch["query_embeddings"]  # (B, E)
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
		y_hat = y_hat.squeeze()
		y = batch["gt_label"]
		loss = self.loss_fn(y_hat, y)
		self.log("val_loss", loss)

		self.val_accuracy(y_hat, y)
		self.val_f1(y_hat, y)
		self.val_recall(y_hat, y)
		self.val_precision(y_hat, y)

		return loss

	def on_validation_epoch_end(self):
		# Compute final metrics across all validation batches
		epoch_accuracy = self.val_accuracy.compute()
		epoch_f1 = self.val_f1.compute()
		epoch_recall = self.val_recall.compute()
		epoch_precision = self.val_precision.compute()

		# Log epoch-level metrics
		self.log("val_accuracy_epoch", epoch_accuracy, prog_bar=True)
		self.log("val_f1_epoch", epoch_f1, prog_bar=True)
		self.log("val_recall_epoch", epoch_recall)
		self.log("val_precision_epoch", epoch_precision)

		# Reset metrics for next epoch
		self.val_accuracy.reset()
		self.val_f1.reset()
		self.val_recall.reset()
		self.val_precision.reset()

	def test_step(self, batch, batch_idx):
		query_embedding = batch["query_embeddings"]  # (B, E)
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
		y_hat = y_hat.squeeze()
		y = batch["gt_label"]
		loss = self.loss_fn(y_hat, y)
		self.log("test_loss", loss)

		self.test_accuracy(y_hat, y)
		self.test_f1(y_hat, y)
		self.test_recall(y_hat, y)
		self.test_precision(y_hat, y)

		return loss

	def on_test_epoch_end(self):
		# Compute and log test metrics
		test_accuracy = self.test_accuracy.compute()
		test_f1 = self.test_f1.compute()
		test_recall = self.test_recall.compute()
		test_precision = self.test_precision.compute()

		self.log("test_accuracy", test_accuracy)
		self.log("test_f1", test_f1)
		self.log("test_recall", test_recall)
		self.log("test_precision", test_precision)

		# Reset test metrics
		self.test_accuracy.reset()
		self.test_f1.reset()
		self.test_recall.reset()
		self.test_precision.reset()

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
