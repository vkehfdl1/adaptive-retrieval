import os
import pathlib

import click
import pandas as pd

from src.data.mfar import AutoRAGQADataset
from src.final.test import ModelEvaluator
from src.linear_mse.module import LinearMSEModule


def main(
	model_checkpoint_path: str,
	qa_embedding_df_test_path,
	semantic_retrieval_df_test_path,
	lexical_retrieval_df_test_path,
	save_path: str,
):
	model = LinearMSEModule.load_from_checkpoint(model_checkpoint_path)

	project_dir = str(
		pathlib.Path(semantic_retrieval_df_test_path).parent.parent.parent.parent
	)
	chroma_path = os.path.join(project_dir, "resources", "chroma")
	dataset = AutoRAGQADataset(
		pd.read_parquet(qa_embedding_df_test_path),
		pd.read_parquet(semantic_retrieval_df_test_path),
		pd.read_parquet(lexical_retrieval_df_test_path),
	)

	evaluator = ModelEvaluator(
		model,
		dataset,
		project_dir,
		chroma_path,
		top_k=50,
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
def cli(
	model_checkpoint_path: str,
	qa_embedding_df_test_path,
	semantic_retrieval_df_test_path,
	lexical_retrieval_df_test_path,
	save_path: str,
):
	main(
		model_checkpoint_path,
		qa_embedding_df_test_path,
		semantic_retrieval_df_test_path,
		lexical_retrieval_df_test_path,
		save_path,
	)


if __name__ == "__main__":
	cli()
