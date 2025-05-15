from pathlib import Path

import pandas as pd

from src.data.mfar import AutoRAGQADataset
from src.upper_bound.run import UpperBoundFinder

root_dir = Path(__file__).parent.parent.parent
data_dir = root_dir / "data"
project_dir = root_dir / "projects"


def test_upper_bound_finder():
	qa_df = pd.read_parquet(data_dir / "allganize" / "qa_embeddings.parquet")
	semantic_retrieval_df = pd.read_parquet(
		project_dir
		/ "allganize"
		/ "0"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "1.parquet"
	)
	lexical_retrieval_df = pd.read_parquet(
		project_dir
		/ "allganize"
		/ "0"
		/ "retrieve_node_line"
		/ "retrieval"
		/ "4.parquet"
	)

	dataset = AutoRAGQADataset(
		qa_df=qa_df,
		semantic_retrieval_df=semantic_retrieval_df,
		lexical_retrieval_df=lexical_retrieval_df,
	)

	finder = UpperBoundFinder(
		dataset,
		project_dir=str(project_dir / "allganize"),
		chroma_path=str(project_dir / "allganize" / "resources" / "chroma"),
		top_k=20,
	)

	finder.run(str(root_dir / "allganize_result.parquet"))
