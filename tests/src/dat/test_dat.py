import pytest

from src.dat.dat import DAT


@pytest.fixture
def dat_instance():
	return DAT()


@pytest.mark.asyncio
async def test_dat(dat_instance):
	question = "What is the capital of France?"
	first_reference = "NomaDamas is Great Team"
	second_reference = "Paris is the capital of France."

	score = await dat_instance.calculate_score(
		question,
		first_reference,
		second_reference,
	)

	assert isinstance(score, float)
	assert score == 0.0
