import itertools
from datetime import datetime
from typing import List, Optional

import click
import pytorch_lightning as pl
import torch
from autorag.evaluation import evaluate_retrieval
from autorag.nodes.retrieval import BM25
from autorag.nodes.retrieval.hybrid_cc import fuse_per_query
from autorag.schema.metricinput import MetricInput
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from src.data.mfar import MfarDataModule
from src.model.losses import ContrastiveLoss
from src.model.models import LinearWeights
from src.utils.chroma import ChromaOnlyEmbeddings


def get_non_duplicate_ids(target_ids, compare_ids) -> List[List[str]]:
	"""
	Get non-duplicate ids from target_ids and compare_ids.
	If you want to non-duplicate ids of semantic_ids, you have to put it at target_ids.
	"""
	result_ids = []
	assert len(target_ids) == len(compare_ids)
	for target_id_list, compare_id_list in zip(target_ids, compare_ids):
		query_duplicated = list(set(compare_id_list) - set(target_id_list))
		duplicate_list = query_duplicated if len(query_duplicated) != 0 else []
		result_ids.append(duplicate_list)
	return result_ids


class MfarTrainingModule(pl.LightningModule):
	def __init__(
		self,
		project_dir: str,
		chroma_path: str,
		collection_name: str = "kure",
		top_k: int = 20,
		temperature: float = 1.0,
		lr: float = 1e-3,
	):
		super().__init__()
		self.chroma_retrieval = ChromaOnlyEmbeddings(chroma_path, collection_name)
		self.bm25_retrieval = BM25(project_dir, bm25_tokenizer="ko_kiwi")
		self.embedding_dimension = 1024
		self.top_k = top_k
		self.temperature = temperature
		self.lr = lr

		self.layer = LinearWeights(self.embedding_dimension)

	def forward(self, x):
		# Input x: [Batch] (string list)
		# Inference
		with torch.no_grad():
			embedding_list = x["query_embeddings"]
			predicted_weight: torch.Tensor = self.layer(embedding_list)

			# Execute Hybrid CC
			semantic_ids, lexical_ids, semantic_scores, lexical_scores = self.retrieve(
				x["query"], x["query_embeddings"].tolist(), retrieval_gt=None
			)
			result_ids, result_scores = self.hybrid_cc_weights(
				predicted_weight,
				semantic_ids,
				lexical_ids,
				semantic_scores,
				lexical_scores,
			)
			return {
				"ids": result_ids,
				"scores": result_scores,
			}

	def training_step(self, batch, batch_idx):
		queries = batch["query"]
		retrieval_gt = batch["retrieval_gt"]
		semantic_ids, lexical_ids, semantic_scores, lexical_scores = self.retrieve(
			queries, batch["query_embeddings"].tolist(), retrieval_gt=retrieval_gt
		)

		predicted_weight = self.layer(batch["query_embeddings"])
		cc_result_ids, cc_result_scores = self.hybrid_cc_weights(
			predicted_weight, semantic_ids, lexical_ids, semantic_scores, lexical_scores
		)

		contrastive_loss = ContrastiveLoss(
			sample_limit=self.top_k, temperature=self.temperature
		)
		loss = contrastive_loss(cc_result_ids, cc_result_scores, retrieval_gt)
		self.log("train/loss", loss.item())
		return loss

	def validation_step(self, batch, batch_idx):
		queries = batch["query"]
		retrieval_gt = batch["retrieval_gt"]
		semantic_ids, lexical_ids, semantic_scores, lexical_scores = self.retrieve(
			queries, batch["query_embeddings"].tolist(), retrieval_gt=retrieval_gt
		)

		with torch.no_grad():
			predicted_weight = self.layer(batch["query_embeddings"])
			cc_result_ids, cc_result_scores = self.hybrid_cc_weights(
				predicted_weight,
				semantic_ids,
				lexical_ids,
				semantic_scores,
				lexical_scores,
			)
			contrastive_loss = ContrastiveLoss(
				sample_limit=self.top_k, temperature=self.temperature
			)
			loss = contrastive_loss(cc_result_ids, cc_result_scores, retrieval_gt)
		self.log("val/loss", loss.item())

		@evaluate_retrieval(
			metric_inputs=list(
				map(
					lambda x: MetricInput(retrieval_gt=x),
					retrieval_gt,
				)
			),
			metrics=[
				"retrieval_f1",
				"retrieval_recall",
				"retrieval_precision",
				"retrieval_ndcg",
				"retrieval_map",
				"retrieval_mrr",
			],
		)
		def calculate_metrics():
			return (
				["" for _ in range(len(cc_result_ids))],
				cc_result_ids,
				cc_result_scores,
			)

		evaluate_result = calculate_metrics()
		self.log("f1", evaluate_result["retrieval_f1"].mean())
		self.log("recall", evaluate_result["retrieval_recall"].mean())
		self.log("precision", evaluate_result["retrieval_precision"].mean())
		self.log("ndcg", evaluate_result["retrieval_ndcg"].mean())
		self.log("map", evaluate_result["retrieval_map"].mean())
		self.log("mrr", evaluate_result["retrieval_mrr"].mean())

	def hybrid_cc_weights(
		self,
		weights: torch.Tensor,
		semantic_ids,
		lexical_ids,
		semantic_scores,
		lexical_scores,
	):
		result_ids, result_scores = [], []
		for idx, weight in enumerate(weights.tolist()):
			result_id_list, result_score_list = fuse_per_query(
				semantic_ids[idx],
				lexical_ids[idx],
				semantic_scores[idx],
				lexical_scores[idx],
				top_k=self.top_k,
				weight=weight,
				normalize_method="tmm",
				semantic_theoretical_min_value=-1.0,
				lexical_theoretical_min_value=0.0,
			)
			result_ids.append(result_id_list)
			result_scores.append(result_score_list)
		return result_ids, result_scores

	def configure_optimizers(self) -> OptimizerLRScheduler:
		optimizer = torch.optim.Adam(
			self.layer.parameters(), self.lr, betas=(0.9, 0.999)
		)
		return [optimizer]

	def retrieve(
		self,
		queries: List[str],
		query_embeddings: List[List[float]],
		retrieval_gt: Optional[List[List]] = None,
	):
		"""The result can be `unsorted`."""
		# Several input batch
		input_queries = list(map(lambda x: [x], queries))
		semantic_ids, semantic_scores = self.chroma_retrieval.query(
			query_embeddings, self.top_k
		)
		lexical_ids, lexical_scores = self.bm25_retrieval._pure(
			input_queries, self.top_k
		)

		lexical_target_ids = get_non_duplicate_ids(lexical_ids, semantic_ids)
		semantic_target_ids = get_non_duplicate_ids(semantic_ids, lexical_ids)

		new_semantic_ids, new_semantic_scores = self.chroma_retrieval.get_ids_scores(
			query_embeddings, semantic_target_ids
		)
		new_lexical_ids, new_lexical_scores = self.bm25_retrieval._pure(
			input_queries, self.top_k, ids=lexical_target_ids
		)

		output_semantic_ids = list(
			map(lambda x, y: x + y, semantic_ids, new_semantic_ids)
		)
		output_lexical_ids = list(map(lambda x, y: x + y, lexical_ids, new_lexical_ids))
		output_semantic_scores = list(
			map(lambda x, y: x + y, semantic_scores, new_semantic_scores)
		)
		output_lexical_scores = list(
			map(lambda x, y: x + y, lexical_scores, new_lexical_scores)
		)

		assert len(output_semantic_ids) == len(output_semantic_scores)
		assert len(output_lexical_ids) == len(output_lexical_scores)
		assert len(output_semantic_ids) == len(queries)

		if retrieval_gt is not None:
			retrieval_gt = [
				list(itertools.chain.from_iterable(x)) for x in retrieval_gt
			]
			# Add Retrieval gt when there is no positive sample
			for idx in range(len(output_semantic_ids)):
				if len(set(retrieval_gt[idx]) & set(output_semantic_scores[idx])) < 1:
					positive_id = retrieval_gt[idx][0]
					positive_id_list, positive_score_list = (
						self.chroma_retrieval.get_ids_scores(
							[query_embeddings[idx]], [[positive_id]]
						)
					)
					output_semantic_ids[idx].append(positive_id_list[0][0])
					output_semantic_scores[idx].append(positive_score_list[0][0])

			for idx in range(len(output_lexical_ids)):
				if len(set(retrieval_gt[idx]) & set(output_lexical_scores[idx])) < 1:
					positive_id = retrieval_gt[idx][0]
					positive_id_list, positive_score_list = self.bm25_retrieval._pure(
						[input_queries[idx]], self.top_k, ids=[[positive_id]]
					)
					output_lexical_ids[idx].append(positive_id_list[0][0])
					output_lexical_scores[idx].append(positive_score_list[0][0])

		return (
			output_semantic_ids,  # List[List[]]
			output_lexical_ids,
			output_semantic_scores,
			output_lexical_scores,
		)


@click.command()
@click.option(
	"--project_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option("--chroma_path", type=click.Path(exists=True, dir_okay=True))
@click.option(
	"--train_data_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.option(
	"--test_data_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.option(
	"--checkpoint_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
def main(
	project_dir: str,
	chroma_path: str,
	train_data_path: str,
	test_data_path: str,
	checkpoint_path: str,
):
	train_module = MfarTrainingModule(project_dir, chroma_path, temperature=0.7)
	data_module = MfarDataModule(train_data_path, test_data_path, num_workers=9)

	tqdm_cb = TQDMProgressBar(refresh_rate=10)
	ckpt_cb = ModelCheckpoint(
		dirpath=checkpoint_path,
		filename="{epoch:02d}-{val_loss:.2f}",
		save_last=True,
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
	)
	trainer.fit(train_module, data_module)


if __name__ == "__main__":
	main()
