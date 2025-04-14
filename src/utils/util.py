def extract_elements(nested_list, exclude_elements):
	result = []
	for item in nested_list:
		if isinstance(item, list):
			# Recursively process nested lists
			result.extend(extract_elements(item, exclude_elements))
		else:
			# Add the item if it's not in the exclusion list
			if item not in exclude_elements:
				result.append(item)
	return result
