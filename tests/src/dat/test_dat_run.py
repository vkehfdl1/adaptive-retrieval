from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.dat.run import DATBenchmark, MultiHopDATBenchmark
from src.data.mfar import AutoRAGQADataset

root_dir = Path(__file__).parent.parent.parent.parent
data_dir = root_dir / "data"
project_dir = root_dir / "projects"

project_name = "allganize"
qa_df = pd.read_parquet(data_dir / "allganize" / "qa_embeddings.parquet")
semantic_retrieval_df = pd.read_parquet(
	project_dir / project_name / "1" / "retrieve_node_line" / "retrieval" / "0.parquet"
)
lexical_retrieval_df = pd.read_parquet(
	project_dir / project_name / "1" / "retrieve_node_line" / "retrieval" / "1.parquet"
)

dataset = AutoRAGQADataset(
	qa_df=qa_df,
	semantic_retrieval_df=semantic_retrieval_df,
	lexical_retrieval_df=lexical_retrieval_df,
)


def test_dat_run():
	instance = DATBenchmark(
		dataset,
		project_dir=str(project_dir / project_name),
		chroma_path=str(project_dir / project_name / "resources" / "chroma"),
		top_k=50,
	)
	instance.run(str(root_dir / f"{project_name}_dat_result_top_k_50.parquet"))


def test_multi_hop_dat_run():
	instance = MultiHopDATBenchmark(
		dataset,
		project_dir=str(project_dir / project_name),
		chroma_path=str(project_dir / project_name / "resources" / "chroma"),
		top_k=20,
		multi_hop_k=3,
	)
	instance.run(
		str(root_dir / f"{project_name}_multi_hop_dat_result_top_k_20_mhk_3.parquet")
	)


if __name__ == "__main__":
	load_dotenv()
	test_dat_run()
