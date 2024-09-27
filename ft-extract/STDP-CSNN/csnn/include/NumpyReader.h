#ifndef _NUMPY_READER_H
#define _NUMPY_READER_H

#include <iostream>
#include "NumpyStruct.h"


class NumpyReader {

public:
	NumpyReader() = delete;

	static void load(const std::string& filename, NumpyArchive& archive) {
		std::ifstream file(filename, std::ios::in | std::ios::binary);

		if(!file) {
			throw std::runtime_error("Unable to open "+filename);
		}

		while(read_zip(file, archive));

		file.close();
	}

	static NumpyArray load(const std::string& filename) {
		std::ifstream file(filename, std::ios::in | std::ios::binary);

		if(!file) {
			throw std::runtime_error("Unable to open "+filename);
		}

		return read_npy_array(file);
	}


private:
	static bool read_zip(std::ifstream& file, NumpyArchive& archive) {
		uint32_t magic;
		file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
		magic = LittleEndian::convert(magic);

		if(magic == ZIP_FILE_MAGIC) {
			uint16_t min_version;
			uint16_t general_flag;
			uint16_t compression_method;
			uint16_t last_modification_time;
			uint16_t last_modification_date;
			uint32_t crc;
			uint32_t compressed_size;
			uint32_t uncompressed_size;

			file.read(reinterpret_cast<char*>(&min_version), sizeof(uint16_t));
			min_version = LittleEndian::convert(min_version);

			file.read(reinterpret_cast<char*>(&general_flag), sizeof(uint16_t));
			general_flag = LittleEndian::convert(general_flag);

			file.read(reinterpret_cast<char*>(&compression_method), sizeof(uint16_t));
			compression_method = LittleEndian::convert(compression_method);

			file.read(reinterpret_cast<char*>(&last_modification_time), sizeof(uint16_t));
			last_modification_time = LittleEndian::convert(last_modification_time);

			file.read(reinterpret_cast<char*>(&last_modification_date), sizeof(uint16_t));
			last_modification_date = LittleEndian::convert(last_modification_date);

			file.read(reinterpret_cast<char*>(&crc), sizeof(uint32_t));
			crc = LittleEndian::convert(crc);

			file.read(reinterpret_cast<char*>(&compressed_size), sizeof(uint32_t));
			compressed_size = LittleEndian::convert(compressed_size);

			file.read(reinterpret_cast<char*>(&uncompressed_size), sizeof(uint32_t));
			uncompressed_size = LittleEndian::convert(uncompressed_size);

			if(compression_method != 0 || compressed_size != uncompressed_size) {
				throw std::runtime_error("Unsupported compression method");
			}

			uint16_t name_length;
			uint16_t extra_field_length;

			file.read(reinterpret_cast<char*>(&name_length), sizeof(uint16_t));
			name_length = LittleEndian::convert(name_length);

			file.read(reinterpret_cast<char*>(&extra_field_length), sizeof(uint16_t));
			extra_field_length = LittleEndian::convert(extra_field_length);

			char* file_name_bytes = new char[name_length];
			file.read(file_name_bytes, name_length);

			char* extra_length_bytes = new char[extra_field_length];
			file.read(extra_length_bytes, extra_field_length);

			std::string file_name(file_name_bytes, name_length);
			std::cout << "Read " << file_name << "..." << std::endl;

			delete[] file_name_bytes;
			delete[] extra_length_bytes;

			archive.add(file_name, read_npy_array(file));

			return true;

		}
		else if(magic == ZIP_DICTIONNARY_MAGIC) {
			return false;
		}
		else {
			throw std::runtime_error("Bad file format : unknwon zip magic");
		}
	}


	static NumpyArray read_npy_array(std::ifstream& file) {
		uint8_t magic_1;
		char magic_2[5];
		file.read(reinterpret_cast<char*>(&magic_1), sizeof(uint8_t));
		file.read(magic_2, 5);

		if(magic_1 != NPY_MAGIC_1 || std::string(magic_2, 5) != NPY_MAGIC_2) {
			throw std::runtime_error("Bad file format : not a numpy file");
		}

		uint8_t major_version;
		uint8_t minor_version;

		file.read(reinterpret_cast<char*>(&major_version), sizeof(uint8_t));
		file.read(reinterpret_cast<char*>(&minor_version), sizeof(uint8_t));

		uint16_t header_length;

		file.read(reinterpret_cast<char*>(&header_length), sizeof(uint16_t));
		header_length = LittleEndian::convert(header_length);

		char* header_length_bytes = new char[header_length];
		file.read(header_length_bytes, header_length);

		std::string header_str(header_length_bytes, header_length);

		size_t cursor = 0;
		std::unique_ptr<NumpyHeaderObject> header = NumpyHeaderObject::read(header_str, cursor);
		NumpyHeaderMap& cast_header = dynamic_cast<NumpyHeaderMap&>(*header);

		NumpyHeaderTuple& shape = dynamic_cast<NumpyHeaderTuple&>(*cast_header.value().at("shape"));

		std::vector<size_t> dimensions;
		std::transform(std::begin(shape.value()), std::end(shape.value()), std::back_inserter(dimensions), [](const std::unique_ptr<NumpyHeaderObject>& element) {
			return dynamic_cast<NumpyHeaderInt&>(*element).value();
		});

		std::cout << "Shape: [";
		for(size_t i=0; i<dimensions.size(); i++) {
			if(i != 0)
				std::cout << ", ";
			std::cout << dimensions.at(i);
		}
		std::cout << "]" << std::endl;

		NumpyArray array(dimensions);

		std::string descr = dynamic_cast<NumpyHeaderString&>(*cast_header.value().at("descr")).value();

		if(descr == "<f8") {
			char* byte_array = new char[array.size()*sizeof(double)];
			file.read(byte_array, array.size()*sizeof(double));

			for(size_t i=0; i<array.size(); i++)
				array.at_index(i) = LittleEndian::convert(*(reinterpret_cast<double*>(byte_array)+i));

			delete[] byte_array;
		}
		else if(descr == ">f8") {
			char* byte_array = new char[array.size()*sizeof(double)];
			file.read(byte_array, array.size()*sizeof(double));

			for(size_t i=0; i<array.size(); i++)
				array.at_index(i) = BigEndian::convert(*(reinterpret_cast<double*>(byte_array)+i));

			delete[] byte_array;
		}
		else if(descr == "<f4") {
			char* byte_array = new char[array.size()*sizeof(float)];
			file.read(byte_array, array.size()*sizeof(float));

			for(size_t i=0; i<array.size(); i++)
				array.at_index(i) = LittleEndian::convert(*(reinterpret_cast<float*>(byte_array)+i));

			delete[] byte_array;
		}
		else if(descr == ">f4") {
			char* byte_array = new char[array.size()*sizeof(float)];
			file.read(byte_array, array.size()*sizeof(float));

			for(size_t i=0; i<array.size(); i++)
				array.at_index(i) = BigEndian::convert(*(reinterpret_cast<float*>(byte_array)+i));

			delete[] byte_array;
		}
		else if(descr == "<i8") {
			char* byte_array = new char[array.size()*sizeof(int64_t)];
			file.read(byte_array, array.size()*sizeof(int64_t));

			for(size_t i=0; i<array.size(); i++)
				array.at_index(i) = LittleEndian::convert(*(reinterpret_cast<int64_t*>(byte_array)+i));

			delete[] byte_array;
		}
		else {
			throw std::runtime_error("unsupported format: "+descr);
		}

		return array;
	}

};

#endif
