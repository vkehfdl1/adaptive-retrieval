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

		return int(response[0]), int(response[1])  # vector result, bm25 result
