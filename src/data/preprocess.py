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


if __name__ == "__main__":
	main()
