# I have to test the result of the model
import os
import pathlib
from typing import Literal

import click
import joblib
import pandas as pd
import torch
from autorag.nodes.retrieval import BM25
from autorag.utils.util import to_list

from src.data.mfar import AutoRAGQADataset
from src.upper_bound.run import get_non_duplicate_ids, calc_metrics
from src.utils.chroma import ChromaOnlyEmbeddings
from src.utils.hybrid_cc import sort_list_to_cc, hybrid_cc


class LgbmModelEvaluator:
	def __init__(
		self,
		model,
		dataset: AutoRAGQADataset,
		project_dir: str,
		chroma_path: str,
		collection_name: str = "kure",
		top_k: int = 50,
		mode: Literal["classification", "regression"] = "classification",
		k: int = 20,
	):
		self.model = model
		self.dataset = dataset
		self.chroma_retrieval = ChromaOnlyEmbeddings(chroma_path, collection_name)
		self.bm25_retrieval = BM25(project_dir, bm25_tokenizer="ko_kiwi")
		self.top_k = top_k
		self.k = k
		self.mode = mode

	def run(self, save_path: str):
		X_test = pd.DataFrame({"query": self.dataset.qa_df["query"].tolist()})
		for idx in range(self.k):
			X_test[f"{idx}_semantic_score"] = self.dataset.semantic_df[
				"retrieve_scores"
			].apply(lambda x: x[idx])
			X_test[f"{idx}_lexical_score"] = self.dataset.lexical_df[
				"retrieve_scores"
			].apply(lambda x: x[idx] / 100.0)

		X_test.drop(columns=["query"], inplace=True)

		if self.mode == "regression":
			weight_list = self.model.predict(X_test)
		elif self.mode == "classification":
			weight_list = self.model.predict_proba(X_test)[:, 1]
		else:
			raise ValueError("Unknown mode")

		weight_tensor = torch.tensor(weight_list)

		semantic_ids, lexical_ids, semantic_scores, lexical_scores = (
			self.retrieve_non_duplicate()
		)

		cc_ids, semantic_score_tensor, lexical_score_tensor = sort_list_to_cc(
			semantic_ids, lexical_ids, semantic_scores, lexical_scores
		)  # List[Tensor]
		cc_scores = hybrid_cc(
			weight_tensor, semantic_score_tensor, lexical_score_tensor
		)

		result_cc_ids = []
		result_cc_scores = []
		for cc_id, cc_score_tensor in zip(cc_ids, cc_scores):
			cc_score = cc_score_tensor.tolist()
			temp = [
				(_id, score)
				for _id, score in sorted(
					zip(cc_id, cc_score),
					key=lambda pair: pair[1],
					reverse=True,
				)
			]
			temp_id_result, temp_score_result = zip(*temp)
			result_cc_ids.append(list(temp_id_result)[: self.top_k])
			result_cc_scores.append(list(temp_score_result)[: self.top_k])

		# Now we have cc_ids and cc_scores result with sorted version

		# Calculate Metrics
		metric_df = calc_metrics(
			self.dataset.qa_df["retrieval_gt"].tolist(), result_cc_ids, result_cc_scores
		)

		metric_df["weights"] = weight_tensor.tolist()
		metric_df["retrieved_ids"] = result_cc_ids
		metric_df["retrieved_scores"] = result_cc_scores

		metric_df.to_parquet(save_path, index=False)

	def retrieve_non_duplicate(self):
		semantic_ids = self.dataset.semantic_df["retrieved_ids"].tolist()
		lexical_ids = self.dataset.lexical_df["retrieved_ids"].tolist()
		semantic_scores = self.dataset.semantic_df["retrieve_scores"].tolist()
		lexical_scores = self.dataset.lexical_df["retrieve_scores"].tolist()
		lexical_target_ids = get_non_duplicate_ids(lexical_ids, semantic_ids)
		semantic_target_ids = get_non_duplicate_ids(semantic_ids, lexical_ids)

		new_semantic_ids, new_semantic_scores = self.chroma_retrieval.get_ids_scores(
			self.dataset.qa_df["query_embeddings"].tolist(), semantic_target_ids
		)
		new_lexical_ids, new_lexical_scores = self.bm25_retrieval._pure(
			self.dataset.qa_df["query"].tolist(), self.top_k, ids=lexical_target_ids
		)

		output_semantic_ids = list(
			map(lambda x, y: to_list(x) + to_list(y), semantic_ids, new_semantic_ids)
		)
		output_lexical_ids = list(
			map(lambda x, y: to_list(x) + to_list(y), lexical_ids, new_lexical_ids)
		)
		output_semantic_scores = list(
			map(
				lambda x, y: to_list(x) + to_list(y),
				semantic_scores,
				new_semantic_scores,
			)
		)
		output_lexical_scores = list(
			map(
				lambda x, y: to_list(x) + to_list(y), lexical_scores, new_lexical_scores
			)
		)

		assert len(output_semantic_ids) == len(output_semantic_scores)
		assert len(output_lexical_ids) == len(output_lexical_scores)

		return (
			output_semantic_ids,  # List[List[]]
			output_lexical_ids,
			output_semantic_scores,
			output_lexical_scores,
		)


def main(
	model_checkpoint_path: str,
	qa_embedding_df_test_path,
	semantic_retrieval_df_test_path,
	lexical_retrieval_df_test_path,
	save_path: str,
	mode: Literal["classification", "regression"] = "classification",
):
	model = joblib.load(model_checkpoint_path)

	project_dir = str(
		pathlib.Path(semantic_retrieval_df_test_path).parent.parent.parent.parent
	)
	chroma_path = os.path.join(project_dir, "resources", "chroma")
	dataset = AutoRAGQADataset(
		pd.read_parquet(qa_embedding_df_test_path),
		pd.read_parquet(semantic_retrieval_df_test_path),
		pd.read_parquet(lexical_retrieval_df_test_path),
	)

	evaluator = LgbmModelEvaluator(
		model,
		dataset,
		project_dir,
		chroma_path,
		top_k=50,
		mode=mode,
		k=20,
	)
	evaluator.run(save_path)


@click.command()
@click.option(
	"--model_checkpoint_path",
	type=str,
	required=True,
	help="Path to the trained model checkpoint",
)
@click.option(
	"--qa_embedding_df_test_path",
	type=str,
	required=True,
	help="Path to the QA test embeddings dataframe",
)
@click.option(
	"--semantic_retrieval_df_test_path",
	type=str,
	required=True,
	help="Path to the semantic retrieval test dataframe",
)
@click.option(
	"--lexical_retrieval_df_test_path",
	type=str,
	required=True,
	help="Path to the lexical retrieval test dataframe",
)
@click.option(
	"--save_path", type=str, required=True, help="Path to save the evaluation results"
)
@click.option(
	"--mode",
	type=click.Choice(["classification", "regression"]),
	default="classification",
)
def cli(
	model_checkpoint_path: str,
	qa_embedding_df_test_path,
	semantic_retrieval_df_test_path,
	lexical_retrieval_df_test_path,
	save_path: str,
	mode,
):
	main(
		model_checkpoint_path,
		qa_embedding_df_test_path,
		semantic_retrieval_df_test_path,
		lexical_retrieval_df_test_path,
		save_path,
		mode,
	)


if __name__ == "__main__":
	cli()
