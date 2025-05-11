import click
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@click.command()
@click.option("--qa_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--save_path", type=click.Path(exists=False, dir_okay=False))
@click.option("--batch_size", type=int, default=64)
def main(qa_path, save_path, batch_size: int = 64):
	assert qa_path.endswith(".parquet")
	assert save_path.endswith(".parquet")

	qa_df = pd.read_parquet(qa_path, engine="pyarrow")
	qa_df["query"] = qa_df["query"].map(lambda x: x[0] if not isinstance(x, str) else x)

	embedding_model = HuggingFaceEmbedding(
		model_name="nlpai-lab/KURE-v1",
		embed_batch_size=batch_size,
	)

	embedding_list = embedding_model.get_text_embedding_batch(
		qa_df["query"].tolist(), show_progress=True
	)

	qa_df["query_embeddings"] = embedding_list

	result_df = qa_df[["qid", "query", "query_embeddings", "retrieval_gt"]]
	result_df.to_parquet(save_path)


if __name__ == "__main__":
	main()
