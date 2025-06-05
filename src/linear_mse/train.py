import os
from datetime import datetime

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.final.data import UpperBoundDataModuleWOTest
from src.linear_mse.module import LinearMSEModule


def main(
	qa_embedding_df_train_path,
	semantic_retrieval_df_train_path,
	lexical_retrieval_df_train_path,
	upper_bound_df_train_path,
	checkpoint_path: str,
):
	train_module = LinearMSEModule(1024)
	data_module = UpperBoundDataModuleWOTest(
		qa_embedding_df_train_path,
		semantic_retrieval_df_train_path,
		lexical_retrieval_df_train_path,
		upper_bound_df_train_path,
		num_workers=9,
	)

	tqdm_cb = TQDMProgressBar(refresh_rate=10)
	resume_ckpt = None
	if checkpoint_path.endswith(".ckpt"):
		resume_ckpt = checkpoint_path
		ckpt_dir = os.path.dirname(checkpoint_path)
	else:
		ckpt_dir = checkpoint_path
	ckpt_cb = ModelCheckpoint(
		dirpath=ckpt_dir,
		filename="{epoch:02d}-{step}-{val_loss:.2f}",
		save_last=True,
		every_n_epochs=1,
	)
	wandb_logger = WandbLogger(name=str(datetime.now()), project="adaptive-retrieval")

	early_stop_callback = EarlyStopping(
		monitor="val_loss", min_delta=0.00, patience=4, verbose=False, mode="min"
	)

	trainer = pl.Trainer(
		accelerator="auto",
		max_epochs=5000,
		log_every_n_steps=48,
		logger=wandb_logger,
		callbacks=[tqdm_cb, ckpt_cb, early_stop_callback],
		check_val_every_n_epoch=1,
		precision="32",
	)
	trainer.fit(train_module, data_module, ckpt_path=resume_ckpt)


@click.command()
@click.option(
	"--qa_embedding_df_train_path",
	type=click.Path(exists=True, file_okay=True),
	required=True,
	help="Path to QA embedding training data",
)
@click.option(
	"--semantic_retrieval_df_train_path",
	type=click.Path(exists=True, file_okay=True),
	required=True,
	help="Path to semantic retrieval training data",
)
@click.option(
	"--lexical_retrieval_df_train_path",
	type=click.Path(exists=True, file_okay=True),
	required=True,
	help="Path to lexical retrieval training data",
)
@click.option(
	"--upper_bound_df_train_path",
	type=click.Path(exists=True, file_okay=True),
	required=True,
	help="Path to upper bound training data",
)
@click.option(
	"--checkpoint_path",
	type=click.Path(exists=True, dir_okay=True, file_okay=True),
	required=True,
	help="Directory to save model checkpoints",
)
def cli(
	qa_embedding_df_train_path,
	semantic_retrieval_df_train_path,
	lexical_retrieval_df_train_path,
	upper_bound_df_train_path,
	checkpoint_path: str,
):
	main(
		qa_embedding_df_train_path,
		semantic_retrieval_df_train_path,
		lexical_retrieval_df_train_path,
		upper_bound_df_train_path,
		checkpoint_path,
	)


if __name__ == "__main__":
	cli()
