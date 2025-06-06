from typing import List, Tuple

from llama_index.llms.openai import OpenAI

from src.dat.prompt import DAT_PROMPT


class DAT:
	def __init__(self, model_name: str = "gpt-4.1-mini"):
		self.llm = OpenAI(model=model_name)

	async def calculate_score(self, question, vector_passage, bm25_passage):
		prompt = DAT_PROMPT.format(
			question=question,
			vector_reference=vector_passage,
			bm25_reference=bm25_passage,
		)
		response = await self.llm.acomplete(prompt)

		response = response.text.strip().split(" ")
		assert len(response) == 2, "The llm response have to have two elements"

		vector_score = int(response[0])
		bm25_score = int(response[1])

		if vector_score == 5 and bm25_score == 5:
			return 0.5
		elif vector_score == 5 and bm25_score != 5:
			return 1.0
		elif vector_score != 5 and bm25_score == 5:
			return 0.0
		elif vector_score == 0 and bm25_score == 0:
			return 0.5
		else:
			return vector_score / (vector_score + bm25_score)


class MultiHopDAT:
	def __init__(self, model_name: str = "gpt-4.1-mini"):
		self.llm = OpenAI(model=model_name)

	async def __ask_to_llm(
		self, question, vector_passage, bm25_passage
	) -> Tuple[int, int]:
		prompt = DAT_PROMPT.format(
			question=question,
			vector_reference=vector_passage,
			bm25_reference=bm25_passage,
		)
		response = await self.llm.acomplete(prompt)

		response = response.text.strip().split(" ")
		assert len(response) == 2, "The llm response have to have two elements"

		vector_score = int(response[0])
		bm25_score = int(response[1])

		return vector_score, bm25_score

	async def calculate_score(
		self, question, vector_passages: List[str], bm25_passages: List[str]
	):
		# First passage
		first_vector_score, first_bm25_score = await self.__ask_to_llm(
			question, vector_passages[0], bm25_passages[0]
		)

		if first_vector_score == 0 and first_bm25_score == 0:
			return 0.5
		elif first_vector_score == 5 and first_bm25_score != 5:
			return 1.0
		elif first_vector_score != 5 and first_bm25_score == 5:
			return 0.0
		else:
			# calculate the rest of the scores
			vec_scores, bm25_scores = [first_vector_score], [first_bm25_score]
			for i in range(1, len(bm25_passages)):
				vec_score, bm25_score = await self.__ask_to_llm(
					question, vector_passages[i], bm25_passages[i]
				)
				vec_scores.append(vec_score)
				bm25_scores.append(bm25_score)

			if sum(vec_scores) == 0 and sum(bm25_scores) == 0:
				return 0.5
			else:
				return sum(vec_scores) / (sum(vec_scores) + sum(bm25_scores))
