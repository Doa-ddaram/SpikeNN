#include "NumpyReader.h"

std::unique_ptr<NumpyHeaderObject> NumpyHeaderObject::read(const std::string& str, size_t& cursor) {
	size_t current_cursor = cursor;

	while(current_cursor < str.size() && std::isspace(str.at(current_cursor)))
		current_cursor++;

	if(current_cursor == str.size()) {
		throw std::runtime_error("End of input");
	}

	std::unique_ptr<NumpyHeaderObject> object;
	if(object = NumpyHeaderBool::read(str, current_cursor)) {
		cursor = current_cursor;
		return object;
	}
	else if(object = NumpyHeaderInt::read(str, current_cursor)) {
		cursor = current_cursor;
		return object;
	}
	else if(object = NumpyHeaderString::read(str, current_cursor)) {
		cursor = current_cursor;
		return object;
	}
	else if(object = NumpyHeaderTuple::read(str, current_cursor)) {
		cursor = current_cursor;
		return object;
	}
	else if(object = NumpyHeaderMap::read(str, current_cursor)) {
		cursor = current_cursor;
		return object;
	}
	else {
		throw std::runtime_error("Unknown next token: "+NumpyHeaderObject::print_next_token(str, current_cursor));
	}
}
