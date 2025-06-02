import pandas as pd
import torch
from autorag.utils.util import to_list
from torch.utils.data import Dataset, DataLoader, default_collate
import pytorch_lightning as pl


class UpperBoundDataset(Dataset):
	def __init__(
		self,
		qa_embedding_df: pd.DataFrame,
		semantic_retrieval_df: pd.DataFrame,
		lexical_retrieval_df: pd.DataFrame,
		upper_bound_df: pd.DataFrame,
	):
		self.qa_embedding_df = qa_embedding_df
		self.semantic_retrieval_df = semantic_retrieval_df
		self.lexical_retrieval_df = lexical_retrieval_df
		self.upper_bound_df = upper_bound_df

		assert (
			len(self.qa_embedding_df)
			== len(self.semantic_retrieval_df)
			== len(self.lexical_retrieval_df)
			== len(self.upper_bound_df)
		)

	def __len__(self):
		return len(self.qa_embedding_df)

	def __getitem__(self, idx):
		return {
			"query": self.upper_bound_df.iloc[idx]["query"],
			"retrieval_gt": to_list(self.upper_bound_df.iloc[idx]["retrieval_gt"]),
			"query_embeddings": torch.from_numpy(
				self.qa_embedding_df.iloc[idx]["query_embeddings"].copy()
			).float(),
			"semantic_retrieve_scores": torch.from_numpy(
				self.semantic_retrieval_df.iloc[idx]["retrieve_scores"].copy()
			).float(),
			"lexical_retrieve_scores": torch.from_numpy(
				self.lexical_retrieval_df.iloc[idx]["retrieve_scores"].copy()
			).float(),
			"gt_weight": torch.tensor(self.upper_bound_df.iloc[idx]["best_weight"]),
		}


class UpperBoundDataLoader(DataLoader):
	def __init__(self, **kwargs):
		super().__init__(
			collate_fn=self._collate_fn,
			**kwargs,
		)

	def _collate_fn(self, batch):
		queries = list(map(lambda x: x["query"], batch))
		retrieval_gt_list = list(map(lambda x: x["retrieval_gt"], batch))

		result = {
			"query": queries,
			"retrieval_gt": retrieval_gt_list,
		}

		other_data = {}
		for key in batch[0].keys():
			if key not in list(result.keys()):
				other_data[key] = default_collate([item[key] for item in batch])

		result.update(other_data)

		return result


class UpperBoundDataModule(pl.LightningDataModule):
	def __init__(
		self,
		qa_embedding_df_train_path,
		semantic_retrieval_df_train_path,
		lexical_retrieval_df_train_path,
		upper_bound_df_train_path,
		qa_embedding_df_test_path,
		semantic_retrieval_df_test_path,
		lexical_retrieval_df_test_path,
		upper_bound_df_test_path,
		batch_size: int = 32,
		num_workers: int = 0,
	):
		super().__init__()
		self.qa_embedding_df_train_path = qa_embedding_df_train_path
		self.semantic_retrieval_df_train_path = semantic_retrieval_df_train_path
		self.lexical_retrieval_df_train_path = lexical_retrieval_df_train_path
		self.upper_bound_df_train_path = upper_bound_df_train_path
		self.qa_embedding_df_test_path = qa_embedding_df_test_path
		self.semantic_retrieval_df_test_path = semantic_retrieval_df_test_path
		self.lexical_retrieval_df_test_path = lexical_retrieval_df_test_path
		self.upper_bound_df_test_path = upper_bound_df_test_path
		self.batch_size = batch_size
		self.num_workers = num_workers

	def setup(self, stage):
		if stage == "fit":
			qa_embedding_df = pd.read_parquet(
				self.qa_embedding_df_train_path, engine="pyarrow"
			)
			semantic_retrieval_df = pd.read_parquet(
				self.semantic_retrieval_df_train_path, engine="pyarrow"
			)
			lexical_retrieval_df = pd.read_parquet(
				self.lexical_retrieval_df_train_path, engine="pyarrow"
			)
			upper_bound_df = pd.read_parquet(
				self.upper_bound_df_train_path, engine="pyarrow"
			)

			slice_idx = int(len(qa_embedding_df) * 0.85)

			self.train_dataset = UpperBoundDataset(
				qa_embedding_df[:slice_idx].reset_index(drop=True),
				semantic_retrieval_df[:slice_idx].reset_index(drop=True),
				lexical_retrieval_df[:slice_idx].reset_index(drop=True),
				upper_bound_df[:slice_idx].reset_index(drop=True),
			)
			self.valid_dataset = UpperBoundDataset(
				qa_embedding_df[slice_idx:].reset_index(drop=True),
				semantic_retrieval_df[slice_idx:].reset_index(drop=True),
				lexical_retrieval_df[slice_idx:].reset_index(drop=True),
				upper_bound_df[slice_idx:].reset_index(drop=True),
			)

		if stage == "test":
			qa_embedding_df = pd.read_parquet(
				self.qa_embedding_df_test_path, engine="pyarrow"
			)
			semantic_retrieval_df = pd.read_parquet(
				self.semantic_retrieval_df_test_path, engine="pyarrow"
			)
			lexical_retrieval_df = pd.read_parquet(
				self.lexical_retrieval_df_test_path, engine="pyarrow"
			)
			upper_bound_df = pd.read_parquet(
				self.upper_bound_df_test_path, engine="pyarrow"
			)

			self.test_dataset = UpperBoundDataset(
				qa_embedding_df,
				semantic_retrieval_df,
				lexical_retrieval_df,
				upper_bound_df,
			)

	def train_dataloader(self):
		return UpperBoundDataLoader(
			dataset=self.train_dataset,
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
		)

	def val_dataloader(self):
		return UpperBoundDataLoader(
			dataset=self.valid_dataset,
			shuffle=False,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
		)

	def test_dataloader(self):
		return UpperBoundDataLoader(
			dataset=self.test_dataset,
			shuffle=False,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
		)
