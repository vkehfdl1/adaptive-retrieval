from pathlib import Path

import pandas as pd

from src.data.mfar import AutoRAGQADataset
from src.upper_bound.run import UpperBoundFinder

root_dir = Path(__file__).parent.parent.parent
data_dir = root_dir / "data"
project_dir = root_dir / "projects"


def test_upper_bound_finder():
	project_name = "mldr-dev"
	qa_df = pd.read_parquet(data_dir / "mldr" / "qa_dev_embeddings.parquet")
	semantic_retrieval_df = pd.read_parquet(
		project_dir
		/ project_name
		/ "0"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "0.parquet",
	)
	lexical_retrieval_df = pd.read_parquet(
		project_dir
		/ project_name
		/ "0"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "1.parquet",
	)

	dataset = AutoRAGQADataset(
		qa_df=qa_df,
		semantic_retrieval_df=semantic_retrieval_df,
		lexical_retrieval_df=lexical_retrieval_df,
	)

	finder = UpperBoundFinder(
		dataset,
		project_dir=str(project_dir / project_name),
		chroma_path=str(project_dir / project_name / "resources" / "chroma"),
		top_k=50,
	)

	finder.run(str(root_dir / f"{project_name}_upper_bound_top_k_50.parquet"))


if __name__ == "__main__":
	test_upper_bound_finder()
