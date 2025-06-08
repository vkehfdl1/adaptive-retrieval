import asyncio
import os

import pandas as pd
import torch
from autorag.nodes.retrieval import BM25
from autorag.utils.util import process_batch

from src.dat.dat import DAT, MultiHopDAT
from src.data.mfar import AutoRAGQADataset
from src.upper_bound.run import get_non_duplicate_ids, calc_metrics
from src.utils.chroma import ChromaOnlyEmbeddings
from src.utils.hybrid_cc import hybrid_cc, sort_list_to_cc


class DATBenchmark:
	def __init__(
		self,
		dataset: AutoRAGQADataset,
		project_dir: str,
		chroma_path: str,
		collection_name: str = "kure",
		top_k: int = 20,
	):
		self.dataset = dataset
		self.chroma_retrieval = ChromaOnlyEmbeddings(chroma_path, collection_name)
		self.bm25_retrieval = BM25(project_dir, bm25_tokenizer="ko_kiwi")
		self.top_k = top_k
		self.dat = DAT()
		self.corpus_df = pd.read_parquet(
			os.path.join(project_dir, "data", "corpus.parquet")
		)

	def run(self, save_path: str):
		if not save_path.endswith(".parquet"):
			raise ValueError("save_path must end with .parquet")

		tasks = [self._run(self.dataset[idx]) for idx in range(len(self.dataset))]
		loop = asyncio.get_event_loop()
		dat_results = loop.run_until_complete(process_batch(tasks, batch_size=16))
		df = pd.DataFrame(
			{
				"weight": dat_results,
			}
		)
		df.to_parquet(save_path, index=False)

		metric_df = calc_metrics(
			self.dataset.qa_df["retrieval_gt"].tolist(),
			list(map(lambda x: x["id_result"], dat_results)),
			list(map(lambda x: x["score_result"], dat_results)),
		)
		metric_df["query"] = self.dataset.qa_df["query"].tolist()
		metric_df["retrieval_gt"] = self.dataset.qa_df["retrieval_gt"].tolist()
		metric_df["dat_weight"] = list(map(lambda x: x["weight"], dat_results))

		metric_df.to_parquet(save_path, index=False)
		return metric_df

	async def _run(self, batch: dict):
		semantic_ids, lexical_ids, semantic_scores, lexical_scores = (
			self.retrieve_non_duplicate(batch)
		)
		cc_ids, semantic_score_tensor, lexical_score_tensor = sort_list_to_cc(
			semantic_ids, lexical_ids, semantic_scores, lexical_scores
		)
		dat_weight = await self._select_weight(batch)
		cc_scores = hybrid_cc(
			torch.Tensor([dat_weight]),
			semantic_score_tensor,
			lexical_score_tensor,
		)
		cc_score = cc_scores[0].tolist()
		# SORT
		result = [
			(_id, score)
			for _id, score in sorted(
				zip(cc_ids[0], cc_score), key=lambda pair: pair[1], reverse=True
			)
		]
		id_result, score_result = zip(*result)
		id_result = list(id_result)[: self.top_k]
		score_result = list(score_result)[: self.top_k]

		return {
			"weight": dat_weight,
			"id_result": id_result,
			"score_result": score_result,
		}

	async def _select_weight(self, batch: dict):
		query = batch["query"]
		argmax_semantic_idx = torch.argmax(batch["semantic_retrieve_scores"])
		argmax_lexical_idx = torch.argmax(batch["lexical_retrieve_scores"])
		best_semantic_id = batch["semantic_retrieved_ids"][argmax_semantic_idx]
		best_lexical_id = batch["lexical_retrieved_ids"][argmax_lexical_idx]

		passages = [
			self.corpus_df[self.corpus_df["doc_id"] == best_semantic_id].iloc[0][
				"contents"
			],
			self.corpus_df[self.corpus_df["doc_id"] == best_lexical_id].iloc[0][
				"contents"
			],
		]

		dat_score = await self.dat.calculate_score(query, passages[0], passages[1])
		return dat_score

	def retrieve_non_duplicate(self, batch):
		semantic_ids = [batch["semantic_retrieved_ids"]]
		lexical_ids = [batch["lexical_retrieved_ids"]]
		semantic_scores = [batch["semantic_retrieve_scores"].tolist()]
		lexical_scores = [batch["lexical_retrieve_scores"].tolist()]
		lexical_target_ids = get_non_duplicate_ids(lexical_ids, semantic_ids)
		semantic_target_ids = get_non_duplicate_ids(semantic_ids, lexical_ids)

		new_semantic_ids, new_semantic_scores = self.chroma_retrieval.get_ids_scores(
			[batch["query_embeddings"].tolist()], semantic_target_ids
		)
		new_lexical_ids, new_lexical_scores = self.bm25_retrieval._pure(
			[batch["query"]], self.top_k, ids=lexical_target_ids
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
		assert len(output_semantic_ids) == 1

		return (
			output_semantic_ids,  # List[List[]]
			output_lexical_ids,
			output_semantic_scores,
			output_lexical_scores,
		)


class MultiHopDATBenchmark(DATBenchmark):
	def __init__(
		self,
		dataset: AutoRAGQADataset,
		project_dir: str,
		chroma_path: str,
		collection_name: str = "kure",
		top_k: int = 20,
		multi_hop_k: int = 5,
	):
		super().__init__(dataset, project_dir, chroma_path, collection_name, top_k)
		self.dat = MultiHopDAT()
		self.multi_hop_k = multi_hop_k

	async def _select_weight(self, batch: dict):
		query = batch["query"]
		semantic_ids = batch["semantic_retrieved_ids"][: self.multi_hop_k]
		lexical_ids = batch["lexical_retrieved_ids"][: self.multi_hop_k]

		semantic_passages = [
			self.corpus_df[self.corpus_df["doc_id"] == _id].iloc[0]["contents"]
			for _id in semantic_ids
		]
		lexical_passages = [
			self.corpus_df[self.corpus_df["doc_id"] == _id].iloc[0]["contents"]
			for _id in lexical_ids
		]

		dat_score = await self.dat.calculate_score(
			query, semantic_passages, lexical_passages
		)
		return dat_score
