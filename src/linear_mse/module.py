import pytorch_lightning as pl
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.linear_mse.models import LinearWeightsAndBias


class LinearMSEModule(pl.LightningModule):
	def __init__(self, embed_size: int):
		super().__init__()
		self.save_hyperparameters()

		self.model = LinearWeightsAndBias(embed_size)
		self.loss_fn = nn.MSELoss()

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		query_embedding = batch["query_embeddings"]  # (B, E)

		y_hat = self.forward(query_embedding)
		y_hat = y_hat.squeeze()
		y = batch["gt_weight"]
		loss = self.loss_fn(y_hat, y)
		self.log("train_loss", loss)
		return loss

	def validation_step(self, batch, batch_idx):
		query_embedding = batch["query_embeddings"]  # (B, E)

		y_hat = self.forward(query_embedding)
		y_hat = y_hat.squeeze()
		y = batch["gt_weight"]
		loss = self.loss_fn(y_hat, y)
		self.log("val_loss", loss)
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
