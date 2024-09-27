#include "ClassParameter.h"
#include <iostream>

ClassParameter::ClassParameter(const ClassParameter& that) noexcept :
	_registration(that._registration), _name(that._name), _parameters(that._parameters) {
}

ClassParameter::ClassParameter(ClassParameter&& that) noexcept :
	_registration(that._registration), _name(std::move(that._name)), _parameters(std::move(that._parameters)) {
}

ClassParameter::~ClassParameter() {
    for(const auto& entry : _parameters) {
        delete entry.second;
	}
}

const std::string& ClassParameter::name() const {
	return _name;
}

void ClassParameter::set_name(const std::string& str) {
	_name = str;
}

AbstractClassParameterVariable& ClassParameter::abstract_parameter(const std::string& name) {
	auto it = _parameters.find(name);

	if(it == std::end(_parameters)) {
		throw std::runtime_error("Parameter \""+name+"\" not found in "+_registration.factory_name()+"."+_registration.class_name());
	}

	return *it->second;
}

const AbstractClassParameterVariable& ClassParameter::abstract_parameter(const std::string& name) const {
	auto it = _parameters.find(name);

	if(it == std::end(_parameters)) {
		throw std::runtime_error("Parameter \""+name+"\" not found in "+_registration.factory_name()+"."+_registration.class_name());
	}

	return *it->second;
}

bool ClassParameter::has_parameter(const std::string& name) const {
	return _parameters.find(name) != std::end(_parameters);
}

void ClassParameter::load(std::istream& stream) {
	_name = Persistence::load_string(stream);

	uint32_t size;
	stream.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

	for(size_t i=0; i<size; i++) {
		std::string name = Persistence::load_string(stream);

		auto it = _parameters.find(name);

		if(it == std::end(_parameters)) {
			throw std::runtime_error("No parameter "+name+" in "+factory_name()+"."+class_name());
		}
		std::cout << "Load " << name << std::endl;
		it->second->load(stream);
	}
}

void ClassParameter::save(std::ostream& stream) const {
	uint32_t magic = Magic;
	stream.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));

	Persistence::save_string(factory_name(), stream);
	Persistence::save_string(class_name(), stream);
	Persistence::save_string(name(), stream);


	uint32_t size = _parameters.size();
	stream.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
	for(const auto& entry : _parameters) {
		Persistence::save_string(entry.first, stream);
		entry.second->save(stream);
	}
}

void ClassParameter::print_parameters(std::ostream& stream, size_t offset) const {
	std::string prefix1(offset, '\t');
	std::string prefix2(offset+1, '\t');
	stream << _registration.factory_name() << "." << _registration.class_name();
	if(!_name.empty()) {
		stream << " (" << _name << ")";
	}
	stream << " {" << std::endl;

    for(const auto& entry : _parameters) {
        stream << prefix2 << entry.first << ": ";
        entry.second->print(stream, offset);
        stream << std::endl;
	}

	stream << prefix1 << "}";
}

const std::string& ClassParameter::factory_name() const {
	return _registration.factory_name();
}

const std::string& ClassParameter::class_name() const {
	return _registration.class_name();
}

void ClassParameter::_initialize(std::default_random_engine& random_engine) {
    for(const auto& entry : _parameters) {
		if(!entry.second->_initialize(random_engine)) {
			throw std::runtime_error("Uninitialized parameter \""+entry.first+"\" in "+_registration.factory_name()+"."+_registration.class_name());
		}
    }
}

void ClassParameter::_check_name(const std::string& name) const {
    if(_parameters.find(name) != std::end(_parameters)) {
        throw std::runtime_error("Parameter with name \""+name+"\" already exists in "+_registration.factory_name()+"."+_registration.class_name());
	}
}

//
//	AbstractClassParameterVariable
//

AbstractClassParameterVariable::AbstractClassParameterVariable() : _initialized(false) {

}

AbstractClassParameterVariable::~AbstractClassParameterVariable() {

}

void AbstractClassParameterVariable::ensure_initialized(std::default_random_engine& random_engine) {
	if(!_initialize(random_engine)) {
		throw std::runtime_error("Uninitialized variable");
	}
}

//
//	AbstractClassParameterFactory
//

AbstractClassParameterFactory::AbstractClassParameterFactory(const std::string& name) : _name(name) {
	ClassParameterRegistry::add(this);
}

AbstractClassParameterFactory::~AbstractClassParameterFactory() {

}

const std::string& AbstractClassParameterFactory::name() const {
	return _name;
}


//
//	AbstractRegisterClassParameter
//

AbstractRegisterClassParameter::AbstractRegisterClassParameter(const std::string& factory_name, const std::string& class_name) :
	_factory_name(factory_name), _class_name(class_name) {

}

AbstractRegisterClassParameter::~AbstractRegisterClassParameter() {
	ClassParameterRegistry::destruct();
}

const std::string& AbstractRegisterClassParameter::factory_name() const {
	return _factory_name;
}

const std::string& AbstractRegisterClassParameter::class_name() const {
	return _class_name;
}

//
//	ClassParameterRegistry
//

ClassParameterRegistry::ClassParameterRegistry() : _map() {

}

ClassParameterRegistry::~ClassParameterRegistry() {
	for(const auto& entry : _map) {
		delete entry.second;
	}
}

void ClassParameterRegistry::add(AbstractClassParameterFactory* factory) {
	auto it = Singleton<ClassParameterRegistry>::instance()._map.find(factory->name());

	if(it != std::end(Singleton<ClassParameterRegistry>::instance()._map)) {
		std::runtime_error("Already registred factory "+factory->name());
	}

	Singleton<ClassParameterRegistry>::instance()._map.emplace(factory->name(), factory);
}

AbstractClassParameterFactory* ClassParameterRegistry::get(const std::string& name) {
	auto it = Singleton<ClassParameterRegistry>::instance()._map.find(name);

	if(it == std::end(Singleton<ClassParameterRegistry>::instance()._map)) {
		throw std::runtime_error("No factory called \""+name+"\" found");
	}

	return it->second;
}



ClassParameter* ClassParameterRegistry::load(std::istream& stream) {
	uint32_t magic;
	stream.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

	if(magic != ClassParameter::Magic) {
		throw std::runtime_error("Invalid ClassParameter magic");
	}

	std::string factory_name = Persistence::load_string(stream);
	std::string class_name = Persistence::load_string(stream);

	ClassParameter* obj = get(factory_name)->abstract_create(class_name);
	obj->load(stream);

	return obj;
}


void ClassParameterRegistry::_throw_incompatible_type(ClassParameter* obj) {
	throw std::runtime_error("Bad factory type: "+obj->factory_name()+" for object "+obj->class_name());
}
