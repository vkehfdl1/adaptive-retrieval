from datetime import datetime
from typing import Literal

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.final.data import UpperBoundDataModule
from src.final.models import MLPRegression, MLPClassifier


def main(
	qa_embedding_df_train_path,
	semantic_retrieval_df_train_path,
	lexical_retrieval_df_train_path,
	upper_bound_df_train_path,
	qa_embedding_df_test_path,
	semantic_retrieval_df_test_path,
	lexical_retrieval_df_test_path,
	upper_bound_df_test_path,
	checkpoint_path: str,
	mode: Literal["classification", "regression"],
):
	if mode == "classification":
		train_module = MLPClassifier(1024)
	elif mode == "regression":
		train_module = MLPRegression(1024)  # Without score distribution
	data_module = UpperBoundDataModule(
		qa_embedding_df_train_path,
		semantic_retrieval_df_train_path,
		lexical_retrieval_df_train_path,
		upper_bound_df_train_path,
		qa_embedding_df_test_path,
		semantic_retrieval_df_test_path,
		lexical_retrieval_df_test_path,
		upper_bound_df_test_path,
		num_workers=9,
	)

	tqdm_cb = TQDMProgressBar(refresh_rate=10)
	ckpt_cb = ModelCheckpoint(
		dirpath=checkpoint_path,
		filename="{epoch:02d}-{step}-{val_loss:.2f}",
		save_last=True,
		every_n_epochs=1,
	)
	wandb_logger = WandbLogger(name=str(datetime.now()), project="adaptive-retrieval")

	early_stop_callback = EarlyStopping(
		monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
	)

	trainer = pl.Trainer(
		accelerator="auto",
		max_epochs=10,
		log_every_n_steps=8,
		logger=wandb_logger,
		callbacks=[tqdm_cb, ckpt_cb, early_stop_callback],
		check_val_every_n_epoch=1,
		precision="32",
	)
	trainer.fit(train_module, data_module)


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
	"--qa_embedding_df_test_path",
	type=click.Path(exists=True, file_okay=True),
	required=True,
	help="Path to QA embedding test data",
)
@click.option(
	"--semantic_retrieval_df_test_path",
	type=click.Path(exists=True, file_okay=True),
	required=True,
	help="Path to semantic retrieval test data",
)
@click.option(
	"--lexical_retrieval_df_test_path",
	type=click.Path(exists=True, file_okay=True),
	required=True,
	help="Path to lexical retrieval test data",
)
@click.option(
	"--upper_bound_df_test_path",
	type=click.Path(exists=True, file_okay=True),
	required=True,
	help="Path to upper bound test data",
)
@click.option(
	"--checkpoint_path",
	type=click.Path(exists=True, dir_okay=True, file_okay=False),
	required=True,
	help="Directory to save model checkpoints",
)
@click.option(
	"--mode",
	type=click.Choice(["classification", "regression"]),
	default="classification",
)
def cli(
	qa_embedding_df_train_path,
	semantic_retrieval_df_train_path,
	lexical_retrieval_df_train_path,
	upper_bound_df_train_path,
	qa_embedding_df_test_path,
	semantic_retrieval_df_test_path,
	lexical_retrieval_df_test_path,
	upper_bound_df_test_path,
	checkpoint_path: str,
	mode,
):
	main(
		qa_embedding_df_train_path,
		semantic_retrieval_df_train_path,
		lexical_retrieval_df_train_path,
		upper_bound_df_train_path,
		qa_embedding_df_test_path,
		semantic_retrieval_df_test_path,
		lexical_retrieval_df_test_path,
		upper_bound_df_test_path,
		checkpoint_path,
		mode,
	)


if __name__ == "__main__":
	cli()
