from autorag.utils import (
	validate_corpus_dataset,
	validate_qa_dataset,
	validate_qa_from_corpus_dataset,
)

from src.data.preprocess import preprocess_mldr, preprocess_mr_tydi


def test_preprocess_mldr():
	corpus, train_qa, dev_qa, test_qa = preprocess_mldr()

	validate_corpus_dataset(corpus)
	validate_qa_dataset(train_qa)
	validate_qa_dataset(dev_qa)
	validate_qa_dataset(test_qa)

	validate_qa_from_corpus_dataset(train_qa, corpus)
	validate_qa_from_corpus_dataset(dev_qa, corpus)
	validate_qa_from_corpus_dataset(test_qa, corpus)


def test_preprocess_mr_tydi():
	corpus, train_qa, dev_qa, test_qa = preprocess_mr_tydi()

	validate_corpus_dataset(corpus)
	validate_qa_dataset(train_qa)
	validate_qa_dataset(dev_qa)
	validate_qa_dataset(test_qa)

	validate_qa_from_corpus_dataset(train_qa, corpus)
	validate_qa_from_corpus_dataset(dev_qa, corpus)
	validate_qa_from_corpus_dataset(test_qa, corpus)
