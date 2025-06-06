import pytest

from src.dat.dat import DAT, MultiHopDAT


@pytest.fixture
def dat_instance():
	return DAT()


@pytest.fixture
def multi_hop_dat_instance():
	return MultiHopDAT()


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


@pytest.mark.asyncio
async def test_multiple_dat(multi_hop_dat_instance):
	question = "What caused the Roman Empire's economic decline, and how did this impact Byzantine military strategies?"
	vec_references = [
		"The Roman Empire had a complex social hierarchy with senators, citizens, and slaves.",
		"The Roman Empire's economic decline was primarily caused by debasement of currency and excessive military spending that drained the treasury.",
		"The Byzantine Empire, facing the economic legacy of Roman fiscal collapse, developed defensive military strategies that relied on fortified positions rather than expensive offensive campaigns.",
	]
	bm25_references = [
		"Byzantine armies used heavy cavalry units called cataphracts in their military formations.",
		"The Mediterranean Sea contains approximately 4 million cubic kilometers of water.",
		"Modern Italy produces over 50 million tons of pasta annually for global export.",
	]

	score = await multi_hop_dat_instance.calculate_score(
		question, vec_references, bm25_references
	)

	assert isinstance(score, float)
	assert score > 0.5
