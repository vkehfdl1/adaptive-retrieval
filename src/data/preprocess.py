# Preprocess all datasets at once
# Allganize RAG evaluation - Already preprocessed by AutoRAG team
# https://huggingface.co/datasets/Shitao/MLDR - MLDR
# Mr.Tydi
# Ko-StrategyQA

import os
import pathlib
from datetime import datetime

import click
import pandas as pd
from datasets import load_dataset

from src.utils.util import extract_elements

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
data_dir = os.path.join(root_dir, "data")


def preprocess_ko_strategyqa_evidence(evidence):
	result = []
	for elem in evidence:
		extracted = extract_elements(elem, ["no_evidence", "operation"])
		if len(extracted) > 0:
			result.append(extracted)
	return result


def preprocess_ko_strategyqa():
	# Corpus preprocess
	corpus = pd.read_parquet(
		os.path.join(data_dir, "ko-strategyqa", "ko-strategy-qa_paragraphs.parquet"),
		engine="pyarrow",
	)
	corpus.drop(columns=["title", "content"], inplace=True)
	corpus.rename(columns={"key": "doc_id", "ko-content": "contents"}, inplace=True)
	corpus["metadata"] = corpus.apply(
		lambda x: {
			"last_modified_datetime": datetime.now(),
			"prev_id": None,
			"next_id": None,
		},
		axis=1,
	)

	# QA preprocess
	def preprocess_qa(path: str):
		df = pd.read_json(path)
		df = df.transpose()
		df["retrieval_gt"] = df["evidence"].apply(preprocess_ko_strategyqa_evidence)
		df["generation_gt"] = df["answer"].apply(lambda x: [x])
		result_df = df[["retrieval_gt", "generation_gt", "question"]]
		result_df = result_df.rename_axis("qid").reset_index()
		result_df.rename(columns={"question": "query"}, inplace=True)
		result_df = result_df[result_df["retrieval_gt"].map(lambda x: len(x)) != 0]
		return result_df

	qa_train = preprocess_qa(
		os.path.join(data_dir, "ko-strategyqa", "ko-strategy-qa_train.json")
	)
	qa_dev = preprocess_qa(
		os.path.join(data_dir, "ko-strategyqa", "ko-strategy-qa_dev.json")
	)

	return corpus, qa_train, qa_dev


def preprocess_mldr():
	corpus = load_dataset("Shitao/MLDR", "corpus-ko", split="corpus").to_pandas()
	# Preprocess Corpus
	corpus.reset_index(drop=True, inplace=True)
	corpus.rename(
		columns={
			"docid": "doc_id",
			"text": "contents",
		},
		inplace=True,
	)
	corpus["metadata"] = corpus["doc_id"].apply(
		lambda x: {
			"last_modified_datetime": datetime.now(),
			"prev_id": None,
			"next_id": None,
		}
	)

	train_qa = load_dataset("Shitao/MLDR", "ko", split="train").to_pandas()
	dev_qa = load_dataset("Shitao/MLDR", "ko", split="dev").to_pandas()
	test_qa = load_dataset("Shitao/MLDR", "ko", split="test").to_pandas()

	# Need to add positive samples (or negative samples) to the corpus if there is no id.
	def add_to_corpus(corpus_df: pd.DataFrame, qa_df: pd.DataFrame) -> pd.DataFrame:
		qa_df["positive_doc_ids"] = qa_df["positive_passages"].apply(
			lambda x: [elem["docid"] for elem in x]
		)
		qa_df["positive_doc_contents"] = qa_df["positive_passages"].apply(
			lambda x: [elem["text"] for elem in x]
		)
		qa_df_exploded = qa_df.explode("positive_doc_ids", ignore_index=True).explode(
			"positive_doc_contents", ignore_index=True
		)

		qa_df_filtered = qa_df_exploded[
			~qa_df_exploded["positive_doc_ids"].isin(corpus_df["doc_id"])
		]
		qa_df_filtered.rename(
			columns={"positive_doc_ids": "doc_id", "positive_doc_contents": "contents"},
			inplace=True,
		)
		qa_df_filtered["metadata"] = qa_df_filtered["doc_id"].apply(
			lambda x: {
				"last_modified_datetime": datetime.now(),
				"prev_id": None,
				"next_id": None,
			}
		)
		corpus_df = pd.concat(
			[corpus_df, qa_df_filtered[["doc_id", "contents", "metadata"]]]
		)
		return corpus_df

	corpus = add_to_corpus(corpus, train_qa)
	corpus = add_to_corpus(corpus, dev_qa)
	corpus = add_to_corpus(corpus, test_qa)

	def preprocess_qa(df: pd.DataFrame):
		df.reset_index(drop=True, inplace=True)

		df.rename(columns={"query_id": "qid"}, inplace=True)
		df["retrieval_gt"] = df["positive_passages"].apply(
			lambda x: [[elem["docid"]] for elem in x]
		)
		df.drop(columns=["positive_passages", "negative_passages"], inplace=True)
		df["generation_gt"] = df["qid"].apply(lambda x: [""])

		return df[["qid", "query", "retrieval_gt", "generation_gt"]]

	return (
		corpus,
		preprocess_qa(train_qa),
		preprocess_qa(dev_qa),
		preprocess_qa(test_qa),
	)


@click.command()
@click.option("--save_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
	"--dataset_name",
	type=click.Choice(["ko-strategyqa", "mr-tydi", "mldr"]),
	required=True,
)
def main(save_dir: str, dataset_name: str):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	if dataset_name == "ko-strategyqa":
		corpus, qa_train, qa_dev = preprocess_ko_strategyqa()
		corpus.to_parquet(os.path.join(save_dir, "corpus.parquet"), index=False)
		qa_train.to_parquet(os.path.join(save_dir, "qa_train.parquet"), index=False)
		qa_dev.to_parquet(os.path.join(save_dir, "qa_dev.parquet"), index=False)
	elif dataset_name == "mldr":
		corpus = preprocess_mldr()


if __name__ == "__main__":
	main()
