from src.utils.util import extract_elements


class TestExtractElements:
	def test_simple_list_no_exclusions(self):
		"""Test extraction from a simple list with no exclusions."""
		input_list = [1, 2, 3, 4, 5]
		exclude = []
		expected = [1, 2, 3, 4, 5]
		assert extract_elements(input_list, exclude) == expected

	def test_simple_list_with_exclusions(self):
		"""Test extraction from a simple list with some exclusions."""
		input_list = [1, 2, 3, 4, 5]
		exclude = [2, 4]
		expected = [1, 3, 5]
		assert extract_elements(input_list, exclude) == expected

	def test_nested_list_no_exclusions(self):
		"""Test extraction from a nested list with no exclusions."""
		input_list = [1, [2, 3], 4, [5, [6, 7]]]
		exclude = []
		expected = [1, 2, 3, 4, 5, 6, 7]
		assert extract_elements(input_list, exclude) == expected

	def test_nested_list_with_exclusions(self):
		"""Test extraction from a nested list with some exclusions."""
		input_list = [1, [2, 3], 4, [5, [6, 7]]]
		exclude = [3, 6]
		expected = [1, 2, 4, 5, 7]
		assert extract_elements(input_list, exclude) == expected

	def test_deeply_nested_list(self):
		"""Test extraction from a deeply nested list."""
		input_list = [1, [2, [3, [4, [5]]]], 6]
		exclude = [2, 5]
		expected = [1, 3, 4, 6]
		assert extract_elements(input_list, exclude) == expected

	def test_mixed_data_types(self):
		"""Test extraction with mixed data types."""
		input_list = ["apple", [10, "banana"], 3.14, ["orange", [True, None]]]
		exclude = ["banana", 3.14, None]
		expected = ["apple", 10, "orange", True]
		assert extract_elements(input_list, exclude) == expected

	def test_empty_list(self):
		"""Test extraction from an empty list."""
		input_list = []
		exclude = [1, 2, 3]
		expected = []
		assert extract_elements(input_list, exclude) == expected

	def test_only_nested_empty_lists(self):
		"""Test extraction from nested empty lists."""
		input_list = [[], [[]], [[], []]]
		exclude = []
		expected = []
		assert extract_elements(input_list, exclude) == expected

	def test_duplicate_elements(self):
		"""Test extraction with duplicate elements in the input list."""
		input_list = [1, 2, 1, [3, 1, 4], [1, [5, 1]]]
		exclude = [4]
		expected = [1, 2, 1, 3, 1, 1, 5, 1]
		assert extract_elements(input_list, exclude) == expected

	def test_exclude_all_elements(self):
		"""Test when all elements are excluded."""
		input_list = [1, 2, [3, 4]]
		exclude = [1, 2, 3, 4]
		expected = []
		assert extract_elements(input_list, exclude) == expected
