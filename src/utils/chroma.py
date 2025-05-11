from typing import List, Tuple

import torch
from autorag.nodes.retrieval.vectordb import get_id_scores
from autorag.utils.util import apply_recursive, to_list
from chromadb import PersistentClient, QueryResult
from chromadb.api.types import IncludeEnum


class ChromaOnlyEmbeddings:
	def __init__(
		self,
		path: str,
		collection_name: str,
		similarity_metric: str = "cosine",
	):
		self.similarity_metric = similarity_metric
		self.client = PersistentClient(path)
		self.collection = self.client.get_or_create_collection(
			name=collection_name,
			metadata={"hnsw:space": similarity_metric},
		)

	def query(
		self, query_embeddings: List[List[float]], top_k: int
	) -> Tuple[List[List[str]], List[List[float]]]:
		if isinstance(query_embeddings, torch.Tensor):
			query_embeddings = query_embeddings.cpu().numpy()
		query_result: QueryResult = self.collection.query(
			query_embeddings=query_embeddings, n_results=top_k
		)
		ids = query_result["ids"]
		scores = query_result["distances"]
		scores = apply_recursive(lambda x: 1 - x, scores)
		return ids, scores

	def fetch(self, ids: List[str]) -> List[List[float]]:
		fetch_result = self.collection.get(ids, include=[IncludeEnum.embeddings])
		return to_list(fetch_result["embeddings"])

	def get_ids_scores(self, query_embeddings: List[List[float]], ids: List[List[str]]):
		if len(ids) < 1:
			return [], []
		if isinstance(query_embeddings, torch.Tensor):
			query_embeddings = query_embeddings.cpu().tolist()
		content_embeddings = [self.fetch(_id) for _id in ids]
		score_result = list(
			map(
				lambda query_embedding_list, content_embedding_list: get_id_scores(
					[query_embedding_list],
					content_embedding_list,
					similarity_metric=self.similarity_metric,
				),
				query_embeddings,
				content_embeddings,
			)
		)
		return ids, score_result
