import pathlib
from typing import Union

import pandas as pd
import pytorch_lightning as pl
import torch
from autorag.utils.util import to_list
from torch.utils.data import Dataset, DataLoader, default_collate


class AutoRAGQADataset(Dataset):
	def __init__(
		self,
		qa_df: pd.DataFrame,
		semantic_retrieval_df: pd.DataFrame,
		lexical_retrieval_df: pd.DataFrame,
	):
		self.qa_df = qa_df
		self.semantic_df = semantic_retrieval_df
		self.lexical_df = lexical_retrieval_df

		assert len(self.qa_df) == len(self.semantic_df) == len(self.lexical_df)

	def __len__(self):
		return len(self.qa_df)

	def __getitem__(self, idx: int):
		qa_row = self.qa_df.iloc[idx]
		return {
			"query": qa_row["query"]
			if isinstance(qa_row["query"], str)
			else qa_row["query"][0],
			"query_embeddings": torch.from_numpy(
				qa_row["query_embeddings"].copy()
			).float(),
			"retrieval_gt": to_list(qa_row["retrieval_gt"]),
			"semantic_retrieved_ids": to_list(
				self.semantic_df.iloc[idx]["retrieved_ids"]
			),
			"semantic_retrieve_scores": torch.from_numpy(
				self.semantic_df.iloc[idx]["retrieve_scores"].copy()
			).float(),
			"lexical_retrieved_ids": to_list(
				self.lexical_df.iloc[idx]["retrieved_ids"]
			),
			"lexical_retrieve_scores": torch.from_numpy(
				self.lexical_df.iloc[idx]["retrieve_scores"].copy()
			).float(),
		}


class AutoRAGDataLoader(DataLoader):
	def __init__(self, **kwargs):
		super().__init__(
			collate_fn=self._collate_fn,
			**kwargs,
		)

	def _collate_fn(self, batch):
		queries = list(map(lambda x: x["query"], batch))
		retrieval_gt_list = list(map(lambda x: x["retrieval_gt"], batch))
		semantic_retrieved_ids = list(map(lambda x: x["semantic_retrieved_ids"], batch))
		lexical_retrieved_ids = list(map(lambda x: x["lexical_retrieved_ids"], batch))
		result = {
			"query": queries,
			"retrieval_gt": retrieval_gt_list,
			"semantic_retrieved_ids": semantic_retrieved_ids,
			"lexical_retrieved_ids": lexical_retrieved_ids,
		}

		other_data = {}
		for key in batch[0].keys():
			if key not in list(result.keys()):
				other_data[key] = default_collate([item[key] for item in batch])

		result.update(other_data)

		return result


class MfarDataModule(pl.LightningDataModule):
	def __init__(
		self,
		qa_train_data_path: Union[pathlib.Path, str],
		semantic_retrieval_df_path: Union[pathlib.Path, str],
		lexical_retrieval_df_path: Union[pathlib.Path, str],
		batch_size: int = 32,
		num_workers: int = 0,
	):
		super().__init__()
		self.qa_train_data_path = qa_train_data_path
		self.semantic_retrieval_df_path = semantic_retrieval_df_path
		self.lexical_retrieval_df_path = lexical_retrieval_df_path
		self.batch_size = batch_size
		self.num_workers = num_workers

	def setup(self, stage):
		if stage == "fit":
			qa_df = pd.read_parquet(self.qa_train_data_path, engine="pyarrow")
			semantic_retrieval_df = pd.read_parquet(
				self.semantic_retrieval_df_path, engine="pyarrow"
			)
			lexical_retrieval_df = pd.read_parquet(
				self.lexical_retrieval_df_path, engine="pyarrow"
			)
			slice_idx = int(len(qa_df) * 0.85)
			self.train_dataset = AutoRAGQADataset(
				qa_df.iloc[:slice_idx].reset_index(drop=True),
				semantic_retrieval_df.iloc[:slice_idx].reset_index(drop=True),
				lexical_retrieval_df.iloc[:slice_idx].reset_index(drop=True),
			)
			self.valid_dataset = AutoRAGQADataset(
				qa_df.iloc[slice_idx:].reset_index(drop=True),
				semantic_retrieval_df.iloc[slice_idx:].reset_index(drop=True),
				lexical_retrieval_df.iloc[slice_idx:].reset_index(drop=True),
			)

	def train_dataloader(self):
		return AutoRAGDataLoader(
			dataset=self.train_dataset,
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
		)

	def val_dataloader(self):
		return AutoRAGDataLoader(
			dataset=self.valid_dataset,
			shuffle=False,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
		)
