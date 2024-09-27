#ifndef _CLASS_PARAMETER_H
#define _CLASS_PARAMETER_H

#include <map>
#include <string>

#include "Tensor.h"
#include "Distribution.h"
#include "Persistence.h"

#define CLASS_PARAMETER_TYPE_PRIMITIVE 0x00
#define CLASS_PARAMETER_TYPE_TENSOR 0x80
#define CLASS_PARAMETER_TYPE_CLASS 0xFF

template<typename T>
class Singleton {

public:
	virtual ~Singleton() {

	}

	static void destruct() {
		delete _instance;
		_instance = nullptr;
	}

	static T& instance() {
		if(_instance == nullptr) {
			_instance = new T;
		}
		return *_instance;
	}

protected:
	Singleton() {

	}

private:
	static T* _instance;
};

template<typename T>
T* Singleton<T>::_instance = nullptr;

template<typename Base>
class AbstractConstructor {

public:
	virtual ~AbstractConstructor() {

	}

	virtual Base* create() = 0;
};

template<typename T, typename Base>
class Constructor : public AbstractConstructor<Base> {

public:
	Constructor() {
		static_assert(std::is_base_of<Base, T>::value, "Incompatible type");
	}

	virtual Base* create() {
		return new T;
	}

};

class ClassParameter;

class AbstractClassParameterFactory {

public:
	AbstractClassParameterFactory(const std::string& name);
	virtual ~AbstractClassParameterFactory();
	const std::string& name() const;

	virtual ClassParameter* abstract_create(const std::string& name) const = 0;

protected:
	std::string _name;
};

template<typename T, typename Factory>
class ClassParameterFactory : public AbstractClassParameterFactory, public Singleton<Factory> {

public:
	typedef T Type;

	ClassParameterFactory(const std::string& name) : AbstractClassParameterFactory(name), _map() {

	}

	virtual ~ClassParameterFactory() {
		for(const auto& entry : _map) {
			delete entry.second;
		}
	}


	template<typename T2>
	static void add(const std::string& name) {
		Singleton<Factory>::instance().template _add<T2>(name);
	}

	static T* create(const std::string& name) {
		return Singleton<Factory>::instance()._create(name);
	}

	virtual ClassParameter* abstract_create(const std::string& name) const {
		return _create(name);
	}

private:
	template<typename T2>
	void _add(const std::string& name) {
		auto it = _map.find(name);

		if(it != std::end(_map)) {
			throw std::runtime_error("Class \""+name+"\" is already registred");
		}

		_map.emplace(name, new Constructor<T2, T>);
	}

	T* _create(const std::string& name) const {
		auto it = _map.find(name);

		if(it == std::end(_map)) {
			throw std::runtime_error("No constructor called \""+name+"\" found in factory \""+_name+"\"");
		}

		return it->second->create();
	}

	std::map<std::string, AbstractConstructor<T>*> _map;

};

class AbstractRegisterClassParameter {

public:
	AbstractRegisterClassParameter(const std::string& factory_name, const std::string& class_name);
	virtual ~AbstractRegisterClassParameter();

	const std::string& factory_name() const;
	const std::string& class_name() const;

private:
	std::string _factory_name;
	std::string _class_name;
};

class ClassParameterRegistry : public Singleton<ClassParameterRegistry> {

public:
	ClassParameterRegistry();
	virtual ~ClassParameterRegistry();

	static void add(AbstractClassParameterFactory* factory);
	static AbstractClassParameterFactory* get(const std::string& name);
	static ClassParameter* load(std::istream& stream);

	template<typename T>
	static T* load_expected(std::istream& stream) {
		ClassParameter* abstract_obj = load(stream);
		T* obj = dynamic_cast<T*>(abstract_obj);

		if(obj == nullptr) {
			_throw_incompatible_type(abstract_obj);
		}

		return obj;
	}

private:
	static void _throw_incompatible_type(ClassParameter* obj);

	std::map<std::string, AbstractClassParameterFactory*> _map;

};

template<typename T, typename Factory>
class RegisterClassParameter : public AbstractRegisterClassParameter {

public:
	RegisterClassParameter(const std::string& name) : AbstractRegisterClassParameter(Factory::instance().name(), name) {
		static_assert(std::is_base_of<typename Factory::Type, T>::value, "Incompatible factory");
		Factory::template add<T>(name);
	}

};


class AbstractClassParameterVariable {

	friend class ClassParameter;

public:
	static constexpr PersistenceType SubClassID = 0xFF;
	static constexpr PersistenceType TensorMask = 0x80;

	AbstractClassParameterVariable();
	virtual ~AbstractClassParameterVariable();

	virtual PersistenceType type_identifier() const = 0;

	void ensure_initialized(std::default_random_engine& random_engine);
	virtual void load(std::istream& stream) = 0;
	virtual void save(std::ostream& stream) const = 0;
	virtual void print(std::ostream& stream, size_t offset) const = 0;

protected:
	virtual bool _initialize(std::default_random_engine& random_engine) = 0;
	bool _initialized;

};

template<typename T>
class PrimitiveClassParameterVariable : public AbstractClassParameterVariable {

public:
	PrimitiveClassParameterVariable(T& ref) : _ref(ref), _distribution(nullptr) {

	}

	PrimitiveClassParameterVariable(const PrimitiveClassParameterVariable<T>& that) = delete;

	virtual ~PrimitiveClassParameterVariable() {
		delete _distribution;
	}

	PrimitiveClassParameterVariable& operator=(const PrimitiveClassParameterVariable<T>& that) = delete;

	virtual PersistenceType type_identifier() const {
		return Persistence::to_indetifier<T>();
	}

	T& get() {
		if(!_initialized) {
			throw std::runtime_error("Variable not initialized");
		}

		return _ref;
	}

	const T& get() const {
		if(!_initialized) {
			throw std::runtime_error("Variable not initialized");
		}

		return _ref;
	}

	void set(T value) {
		_set(new distribution::Constant<T>(value));
	}

	template<template<typename> class DistributionType, typename... Args>
	void distribution(Args&&... args) {
		_set(new DistributionType<T>(std::forward<Args>(args)...));
	}

	virtual void load(std::istream& stream) {
		if(_initialized) {
			throw std::runtime_error("Can't load variable after initialization");
		}

		PersistenceType type;
		stream.read(reinterpret_cast<char*>(&type), sizeof(PersistenceType));

		if(type != Persistence::to_indetifier<T>()) {
			throw std::runtime_error("Incompatible variable type"); // TODO expected str vs actual str
		}

		T value;
		stream.read(reinterpret_cast<char*>(&value), sizeof(T));

		set(value);
	}

	virtual void save(std::ostream& stream) const {
		if(!_initialized) {
			throw std::runtime_error("Variable need to be initialized before save");
		}

		PersistenceType type = Persistence::to_indetifier<T>();
		stream.write(reinterpret_cast<const char*>(&type), sizeof(PersistenceType));
		stream.write(reinterpret_cast<const char*>(&_ref), sizeof(T));
	}

	virtual void print(std::ostream& stream, size_t) const {
		stream << (_initialized ? _distribution->to_string() : "Uninitialized");
	}

protected:
	virtual bool _initialize(std::default_random_engine& random_engine) {
		if(_initialized) {
			return true;
		}

		if(_distribution == nullptr) {
			return false;
		}
		_ref = _distribution->generate(random_engine);
		_initialized = true;
		return true;
	}

private:
	void _set(Distribution<T>* d) {
		if(_initialized) {
			throw std::runtime_error("Variable already initialized");
		}

		delete _distribution;
		_distribution = d;
	}

	T& _ref;
	Distribution<T>* _distribution;
};

template<typename T>
class TensorClassParameterVariable : public AbstractClassParameterVariable {

public:
	TensorClassParameterVariable(Tensor<T>& ref) : _ref(ref), _distribution(nullptr), _value(nullptr), _shape()  {

	}

	TensorClassParameterVariable(const TensorClassParameterVariable<T>& that) = delete;

	virtual ~TensorClassParameterVariable() {
		delete _distribution;
		delete _value;
	}

	TensorClassParameterVariable& operator=(const TensorClassParameterVariable<T>& that) = delete;

	virtual PersistenceType type_identifier() const {
		return Persistence::to_indetifier<T>() | TensorMask;
	}

	void set(const Tensor<T>& value) {
		_set(new Tensor<T>(value));
	}

	Tensor<T>& get() {
		if(!_initialized) {
			throw std::runtime_error("Variable not initialized");
		}

		return _ref;
	}

	const Tensor<T>& get() const {
		if(!_initialized) {
			throw std::runtime_error("Variable not initialized");
		}

		return _ref;
	}

	void fill(T value) {
		_set(new distribution::Constant<T>(value));
	}

	template<template<typename> class DistributionType, typename... Args>
	void distribution(Args&&... args) {
		_set(new DistributionType<T>(std::forward<Args>(args)...));
	}

	void shape(const Shape& s) {
		if(_initialized) {
			throw std::runtime_error("Variable already initialized");
		}
		_shape = s;
	}

	template<typename... Is>
	void shape(Is... is) {
		if(_initialized) {
			throw std::runtime_error("Variable already initialized");
		}
		_shape = Shape({is...});
	}

	virtual void load(std::istream& stream) {
		if(_initialized) {
			throw std::runtime_error("Can't load variable after initialization");
		}
/*
		PersistenceType type;
		stream.read(reinterpret_cast<char*>(&type), sizeof(PersistenceType));

		if(type != (Persistence::to_indetifier<T>() | TensorMask)) {
			throw std::runtime_error("Incompatible variable type"); // TODO expected str vs actual str
		}

		uint32_t number_dim;
		stream.read(reinterpret_cast<char*>(&number_dim), sizeof(uint32_t));

		std::vector<size_t> dims;
		for(size_t i=0; i<number_dim; i++) {
			uint32_t dim;
			stream.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
			dims.push_back(dim);
		}

		Shape s(dims);
		Tensor<T> value(s);
		stream.read(reinterpret_cast<char*>(value.begin()), s.product()*sizeof(T));
*/
		set(Persistence::load_tensor<T>(stream));
	}

	virtual void save(std::ostream& stream) const {
		if(!_initialized) {
			throw std::runtime_error("Variable need to be initialized before save");
		}
/*
		PersistenceType type = Persistence::to_indetifier<T>() | TensorMask;
		stream.write(reinterpret_cast<const char*>(&type), sizeof(PersistenceType));
		uint32_t number_dim = _ref.shape().number();
		stream.write(reinterpret_cast<const char*>(&number_dim), sizeof(uint32_t));
		for(size_t i=0; i<number_dim; i++) {
			uint32_t dim = _ref.shape().dim(i);
			stream.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
		}

		stream.write(reinterpret_cast<const char*>(_ref.begin()), _ref.shape().product()*sizeof(T));*/

		Persistence::save_tensor(_ref, stream);
	}

	virtual void print(std::ostream& stream, size_t) const {
		if(!_initialized) {
			stream << "Uninitialized";
		}
		else if(_distribution != nullptr) {
			stream << _distribution->to_string() << " " << _shape.to_string();
		}
		else {
			stream << " Tensor " << _value->shape().to_string();
		}
	}

protected:
	virtual bool _initialize(std::default_random_engine& random_engine) {
		if(_initialized) {
			return true;
		}

		if((_distribution != nullptr && _shape.number() > 0)) {

			_ref = Tensor<T>(_shape);

			size_t s = _shape.product();

			for(size_t i=0; i<s; i++) {
				_ref.at_index(i) = _distribution->generate(random_engine);
			}

			_initialized = true;

			return true;
		}
		else if(_value != nullptr) {

			_ref = *_value;

			_initialized = true;

			return true;
		}
		else {
			return false;
		}
	}

private:
	void _set(Distribution<T>* d) {
		if(_initialized) {
			throw std::runtime_error("Variable already initialized");
		}


		delete _distribution;
		delete _value;
		_distribution = d;
		_value = nullptr;
	}

	void _set(Tensor<T>* tensor) {
		if(_initialized) {
			throw std::runtime_error("Variable already initialized");
		}


		delete _distribution;
		delete _value;
		_distribution = nullptr;
		_value = tensor;
	}

private:
	Tensor<T>& _ref;
	Distribution<T>* _distribution;
	Tensor<T>* _value;
	Shape _shape;
};

template<typename T>
class SubClassParameterVariable : public AbstractClassParameterVariable {

public:
	SubClassParameterVariable(T*& ref) : _ref(ref) {
		if(ref != nullptr) {
			throw std::runtime_error("ClassParameter should be null");
		}
	}

	~SubClassParameterVariable() {
		delete _ref;
	}

	SubClassParameterVariable(const SubClassParameterVariable<T>& that) = delete;
	SubClassParameterVariable& operator=(const SubClassParameterVariable<T>& that) = delete;

	virtual PersistenceType type_identifier() const {
		return SubClassID;
	}

	template<typename T2, typename... Args>
	typename std::enable_if<std::is_base_of<T, T2>::value>::type set(Args&&... args) {
		if(_initialized) {
			throw std::runtime_error("Variable already initialized");
		}
		_ref = new T2(std::forward<Args>(args)...);
	}

	T& get() {
		if(!_initialized) {
			throw std::runtime_error("Variable already initialized");
		}

		return *_ref;
	}

	const T& get() const {
		if(!_initialized) {
			throw std::runtime_error("Variable already initialized");
		}

		return *_ref;
	}

	virtual void load(std::istream& stream) {
		if(_initialized) {
			throw std::runtime_error("Can't load variable after initialization");
		}

		PersistenceType type;
		stream.read(reinterpret_cast<char*>(&type), sizeof(PersistenceType));

		if(type != SubClassID) {
			throw std::runtime_error("Incompatible variable type (expected subclass)");
		}

		_ref = ClassParameterRegistry::load_expected<T>(stream);
	}

	virtual void save(std::ostream& stream) const {
		if(!_initialized) {
			throw std::runtime_error("Variable need to be initialized before save");
		}

		PersistenceType type = SubClassID;
		stream.write(reinterpret_cast<const char*>(&type), sizeof(PersistenceType));

		_ref->save(stream);
	}

	virtual void print(std::ostream& stream, size_t offset) const {
		if(!_initialized) {
			stream << "Uninitialized";
		}
		else {
			_ref->print_parameters(stream, offset+1);
		}
	}

protected:
	virtual bool _initialize(std::default_random_engine& random_engine) {
		if(_ref != nullptr) {
			_ref->_initialize(random_engine);
			_initialized = true;
			return true;
		}
		else {
			return false;
		}
	}

private:
	T*& _ref;
};

template<typename T>
struct IsTensor : public std::false_type {};


template<typename T>
struct IsTensor<Tensor<T>> : public std::true_type {};

class ClassParameter {

	friend class AbstractExperiment;

	template<typename U>
	friend class SubClassParameterVariable;

public:
	static constexpr uint32_t Magic = 0xC1A55ABC;

	//ClassParameter(const std::string& factory_name, const std::string& class_name);
	template<typename T, typename Factory>
	ClassParameter(const RegisterClassParameter<T, Factory>& registration) :
		_registration(registration), _name(), _parameters() {

	}


	ClassParameter(const ClassParameter& that) noexcept;
	ClassParameter(ClassParameter&& that) noexcept;

	virtual ~ClassParameter();

	ClassParameter& operator=(const ClassParameter& that) noexcept = delete;
	ClassParameter& operator=(ClassParameter&& that) noexcept = delete;

	const std::string& name() const;
	void set_name(const std::string& str);

	template<typename T>
	typename std::enable_if<std::is_arithmetic<T>::value>::type add_parameter(const std::string& name, T& p) {
		_check_name(name);
		_parameters.emplace(name, new PrimitiveClassParameterVariable<T>(p));
	}

	template<typename T>
	typename std::enable_if<std::is_arithmetic<T>::value>::type add_parameter(const std::string& name, T& p, T default_value) {
		_check_name(name);
		PrimitiveClassParameterVariable<T>* obj = new PrimitiveClassParameterVariable<T>(p);
		_parameters.emplace(name, obj);
		obj->set(default_value);
	}


	template<typename T>
	void add_parameter(const std::string& name, Tensor<T>& p) {
		_check_name(name);
		_parameters.emplace(name, new TensorClassParameterVariable<T>(p));
	}

	template<typename T>
	typename std::enable_if<std::is_base_of<ClassParameter, T>::value>::type add_parameter(const std::string& name, T*& p) {
		_check_name(name);
		_parameters.emplace(name, new SubClassParameterVariable<T>(p));
	}

	AbstractClassParameterVariable& abstract_parameter(const std::string& name);
	const AbstractClassParameterVariable& abstract_parameter(const std::string& name) const;

	template<typename T>
	typename std::enable_if<std::is_arithmetic<T>::value, PrimitiveClassParameterVariable<T>&>::type parameter(const std::string& name) {
		return dynamic_cast<PrimitiveClassParameterVariable<T>&>(abstract_parameter(name));
	}

	template<typename T>
	typename std::enable_if<std::is_arithmetic<T>::value, const PrimitiveClassParameterVariable<T>&>::type parameter(const std::string& name) const {
		return dynamic_cast<const PrimitiveClassParameterVariable<T>&>(abstract_parameter(name));
	}

	template<typename T>
	typename std::enable_if<IsTensor<T>::value, TensorClassParameterVariable<typename T::Type>&>::type parameter(const std::string& name) {
		return dynamic_cast<TensorClassParameterVariable<typename T::Type>&>(abstract_parameter(name));
	}

	template<typename T>
	typename std::enable_if<IsTensor<T>::value, const TensorClassParameterVariable<typename T::Type>&>::type parameter(const std::string& name) const {
		return dynamic_cast<const TensorClassParameterVariable<typename T::Type>&>(abstract_parameter(name));
	}

	template<typename T>
	typename std::enable_if<std::is_base_of<ClassParameter, T>::value, SubClassParameterVariable<T>&>::type parameter(const std::string& name) {
		return dynamic_cast<SubClassParameterVariable<T>&>(abstract_parameter(name));
	}

	template<typename T>
	typename std::enable_if<std::is_base_of<ClassParameter, T>::value, const SubClassParameterVariable<T>&>::type parameter(const std::string& name) const {
		return dynamic_cast<const SubClassParameterVariable<T>&>(abstract_parameter(name));
	}

	template<typename T>
	typename std::enable_if<std::is_arithmetic<T>::value, bool>::type is_type(const std::string& name) const {
		return abstract_parameter(name).type_identifier() == Persistence::to_indetifier<T>();
	}

	template<typename T>
	typename std::enable_if<IsTensor<T>::value, bool>::type is_type(const std::string& name) const {
		return abstract_parameter(name).type_identifier() == (AbstractClassParameterVariable::TensorMask | Persistence::to_indetifier<typename T::Type>());
	}

	template<typename T>
	typename std::enable_if<std::is_base_of<ClassParameter, T>::value, bool>::type is_type(const std::string& name) const {
		const AbstractClassParameterVariable& p = abstract_parameter(name);

		if(p.type_identifier() == AbstractClassParameterVariable::SubClassID) {
			return dynamic_cast<SubClassParameterVariable<T>*>(&p) != nullptr;
		}
		else {
			return false;
		}
	}

	bool has_parameter(const std::string& name) const;

	void load(std::istream& stream);
	void save(std::ostream& stream) const;

	void print_parameters(std::ostream& stream, size_t offset = 0) const;

	const std::string& factory_name() const;
	const std::string& class_name() const;


private:
	void _initialize(std::default_random_engine& random_engine);

	void _check_name(const std::string& name) const;

	const AbstractRegisterClassParameter& _registration;

	std::string _name;
	std::map<std::string, AbstractClassParameterVariable*> _parameters;

};

#endif
