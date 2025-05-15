import itertools
from datetime import datetime
from typing import List, Optional

import pytorch_lightning as pl
import torch
from autorag.evaluation import evaluate_retrieval
from autorag.nodes.retrieval import BM25
from autorag.schema.metricinput import MetricInput
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from src.data.mfar import MfarDataModule
from src.model.losses import ContrastiveLoss
from src.model.models import LinearWeights
from src.utils.chroma import ChromaOnlyEmbeddings
from src.utils.hybrid_cc import hybrid_cc, sort_list_to_cc


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
			cc_ids, semantic_score_tensor, lexical_score_tensor = sort_list_to_cc(
				semantic_ids, lexical_ids, semantic_scores, lexical_scores
			)
			cc_scores = hybrid_cc(
				predicted_weight,
				semantic_score_tensor,
				lexical_score_tensor,
			)
			cc_scores = list(map(lambda x: x.tolist(), cc_scores))
			result = [
				(_id, score)
				for score, _id in sorted(
					zip(cc_scores, cc_ids), key=lambda pair: pair[0], reverse=True
				)
			]
			id_result, score_result = zip(*result)
			return {
				"ids": list(map(lambda x: x[: self.top_k], list(id_result))),
				"scores": list(map(lambda x: x[: self.top_k], list(score_result))),
			}

	def training_step(self, batch, batch_idx):
		predicted_weight = self.layer(batch["query_embeddings"])

		semantic_ids, lexical_ids, semantic_scores, lexical_scores = (
			self.retrieve_non_duplicate(batch)
		)
		cc_ids, semantic_score_tensor, lexical_score_tensor = sort_list_to_cc(
			semantic_ids, lexical_ids, semantic_scores, lexical_scores
		)
		cc_scores = hybrid_cc(
			predicted_weight,
			semantic_score_tensor,
			lexical_score_tensor,
		)
		cc_scores = list(map(lambda x: x.tolist(), cc_scores))
		contrastive_loss = ContrastiveLoss(
			sample_limit=self.top_k, temperature=self.temperature
		)
		loss = contrastive_loss(cc_ids, cc_scores, batch["retrieval_gt"])
		self.log("train/loss", loss.item())
		return loss

	def validation_step(self, batch, batch_idx):
		retrieval_gt = batch["retrieval_gt"]

		with torch.no_grad():
			predicted_weight = self.layer(batch["query_embeddings"])
			batch.pop("retrieval_gt")
			semantic_ids, lexical_ids, semantic_scores, lexical_scores = (
				self.retrieve_non_duplicate(batch)
			)
			cc_ids, semantic_score_tensor, lexical_score_tensor = sort_list_to_cc(
				semantic_ids, lexical_ids, semantic_scores, lexical_scores
			)
			cc_scores = hybrid_cc(
				predicted_weight,
				semantic_score_tensor,
				lexical_score_tensor,
			)
			contrastive_loss = ContrastiveLoss(
				sample_limit=self.top_k, temperature=self.temperature
			)
			cc_scores = list(map(lambda x: x.tolist(), cc_scores))
			loss = contrastive_loss(cc_ids, cc_scores, retrieval_gt)
		self.log("val_loss", loss.item())

		result = [
			(_id, score)
			for score, _id in sorted(
				zip(cc_scores, cc_ids), key=lambda pair: pair[0], reverse=True
			)
		]
		id_result, score_result = zip(*result)

		id_result = list(map(lambda x: x[: self.top_k], list(id_result)))
		score_result = list(map(lambda x: x[: self.top_k], list(score_result)))

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
				["" for _ in range(len(id_result))],
				id_result,
				score_result,
			)

		evaluate_result = calculate_metrics()
		self.log("f1", evaluate_result["retrieval_f1"].mean())
		self.log("recall", evaluate_result["retrieval_recall"].mean())
		self.log("precision", evaluate_result["retrieval_precision"].mean())
		self.log("ndcg", evaluate_result["retrieval_ndcg"].mean())
		self.log("map", evaluate_result["retrieval_map"].mean())
		self.log("mrr", evaluate_result["retrieval_mrr"].mean())

	def configure_optimizers(self) -> OptimizerLRScheduler:
		optimizer = torch.optim.Adam(
			self.layer.parameters(), self.lr, betas=(0.8, 0.999)
		)
		return [optimizer]

	def retrieve_non_duplicate(self, batch):
		semantic_ids = batch["semantic_retrieved_ids"]
		lexical_ids = batch["lexical_retrieved_ids"]
		semantic_scores = batch["semantic_retrieve_scores"].tolist()
		lexical_scores = batch["lexical_retrieve_scores"].tolist()
		lexical_target_ids = get_non_duplicate_ids(lexical_ids, semantic_ids)
		semantic_target_ids = get_non_duplicate_ids(semantic_ids, lexical_ids)

		new_semantic_ids, new_semantic_scores = self.chroma_retrieval.get_ids_scores(
			batch["query_embeddings"].tolist(), semantic_target_ids
		)
		new_lexical_ids, new_lexical_scores = self.bm25_retrieval._pure(
			batch["query"], self.top_k, ids=lexical_target_ids
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
		assert len(output_semantic_ids) == len(batch["query"])

		if "retrieval_gt" in batch.keys() and batch["retrieval_gt"] is not None:
			retrieval_gt = batch["retrieval_gt"]
			retrieval_gt = [
				list(itertools.chain.from_iterable(x)) for x in retrieval_gt
			]
			# Add Retrieval gt when there is no positive sample
			for idx in range(len(output_semantic_ids)):
				if len(set(retrieval_gt[idx]) & set(output_semantic_scores[idx])) < 1:
					positive_id = retrieval_gt[idx][0]
					positive_id_list, positive_score_list = (
						self.chroma_retrieval.get_ids_scores(
							[batch["query_embeddings"].tolist()[idx]], [[positive_id]]
						)
					)
					output_semantic_ids[idx].append(positive_id_list[0][0])
					output_semantic_scores[idx].append(positive_score_list[0][0])

			for idx in range(len(output_lexical_ids)):
				if len(set(retrieval_gt[idx]) & set(output_lexical_scores[idx])) < 1:
					positive_id = retrieval_gt[idx][0]
					positive_id_list, positive_score_list = self.bm25_retrieval._pure(
						[batch["query"][idx]], self.top_k, ids=[[positive_id]]
					)
					output_lexical_ids[idx].append(positive_id_list[0][0])
					output_lexical_scores[idx].append(positive_score_list[0][0])

		return (
			output_semantic_ids,  # List[List[]]
			output_lexical_ids,
			output_semantic_scores,
			output_lexical_scores,
		)

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

		return self.retrieve_non_duplicate(
			{
				"semantic_retrieved_ids": semantic_ids,
				"lexical_retrieved_ids": lexical_ids,
				"semantic_retrieve_scores": torch.Tensor(semantic_scores),
				"lexical_retrieve_scores": torch.Tensor(lexical_scores),
				"retrieval_gt": retrieval_gt,
				"query": queries,
				"query_embeddings": torch.Tensor(query_embeddings),
			}
		)


# @click.command()
# @click.option(
# 	"--project_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
# )
# @click.option("--chroma_path", type=click.Path(exists=True, dir_okay=True))
# @click.option(
# 	"--train_data_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
# )
# @click.option(
# 	"--semantic_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
# )
# @click.option(
# 	"--lexical_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
# )
# @click.option(
# 	"--checkpoint_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
# )
def main(
	project_dir: str,
	chroma_path: str,
	train_data_path: str,
	semantic_path: str,
	lexical_path: str,
	checkpoint_path: str,
):
	train_module = MfarTrainingModule(project_dir, chroma_path, temperature=0.7)
	data_module = MfarDataModule(
		train_data_path, semantic_path, lexical_path, num_workers=6
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
	)
	trainer.fit(train_module, data_module)


if __name__ == "__main__":
	from pathlib import Path

	root_dir = Path(__file__).parent.parent.parent
	main(
		project_dir=str(root_dir / "projects" / "ko-strategyqa-dev"),
		chroma_path=str(
			root_dir / "projects" / "ko-strategyqa-dev" / "resources" / "chroma"
		),
		train_data_path=str(
			root_dir / "data" / "ko-strategyqa" / "qa_train_embeddings.parquet"
		),
		semantic_path=str(
			root_dir
			/ "projects"
			/ "ko-strategyqa-train"
			/ "1"
			/ "retrieve_node_line"
			/ "retrieval"
			/ "0.parquet"
		),
		lexical_path=str(
			root_dir
			/ "projects"
			/ "ko-strategyqa-train"
			/ "1"
			/ "retrieve_node_line"
			/ "retrieval"
			/ "1.parquet"
		),
		checkpoint_path=str(root_dir / "train_result"),
	)
	# main()
