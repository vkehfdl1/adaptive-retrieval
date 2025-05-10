import pandas as pd
import pytorch_lightning as pl
from autorag.utils import validate_qa_dataset
from autorag.utils.util import to_list
from torch.utils.data import Dataset, DataLoader


class AutoRAGQADataset(Dataset):
	def __init__(self, df: pd.DataFrame):
		validate_qa_dataset(df)
		self.df = df

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		return {
			"query": row["query"],
			"retrieval_gt": to_list(row["retrieval_gt"]),
		}


class MfarDataModule(pl.LightningDataModule):
	def __init__(
		self,
		qa_train_data_path,
		qa_test_data_path,
		batch_size: int = 32,
	):
		super().__init__()
		self.qa_train_data_path = qa_train_data_path
		self.qa_test_data_path = qa_test_data_path
		self.batch_size = batch_size

	def setup(self, stage):
		if stage == "fit":
			df = pd.read_parquet(self.qa_train_data_path, engine="pyarrow")
			train_df = df.sample(frac=0.85, random_state=42).reset_index(drop=True)
			valid_df = df.drop(train_df.index).reset_index(drop=True)
			self.train_dataset = AutoRAGQADataset(train_df)
			self.valid_dataset = AutoRAGQADataset(valid_df)

		if stage == "test":
			df = pd.read_parquet(self.qa_test_data_path, engine="pyarrow")
			self.test_dataset = AutoRAGQADataset(df)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

	def val_dataloader(self):
		return DataLoader(self.valid_dataset, shuffle=False, batch_size=self.batch_size)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)

	def predict_dataloader(self):
		return self.test_dataloader()
