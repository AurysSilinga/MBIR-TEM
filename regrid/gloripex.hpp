//
// Copyright 2013 by Forschungszentrum Juelich GmbH
//
#ifndef GLORIPEX_HPP
#define GLORIPEX_HPP

// this removes strange linkage issues with extensions covering multiple files
#define PY_ARRAY_UNIQUE_SYMBOL GLORIPEX_PY_ARRAY

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <complex>
#include <boost/python/detail/wrap_python.hpp>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace std {
    /// Conveniencev  vvc ostream operator for std::vector.
    template<typename T>
    inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
        os << "[";
        for (typename std::vector<T>::size_type i = 0; i < vec.size(); ++ i) {
            os << (i == 0 ? "" : ", ") << vec[i];
        }
        os << "]";
        return os;
    }

    /// Convenience ostream operator for std::vector<std::string>.
    template<>
    inline std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec) {
        os << "[";
        for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++ i) {
            os << (i == 0 ? "'" : ", '") << vec[i] << "'";
        }
        os << "]";
        return os;
    }
}


namespace gloripex {
    namespace python = boost::python;

    typedef std::complex<float> complex64;
    typedef std::complex<double> complex128;
    typedef uint16_t cub_t;

    /// Simple struct for error handling.
    ///
    /// This should be the base struct of all gloripy Exceptions. It is
    /// translated to a python runtime error.
    struct Exception : std::exception {
        std::string msg_;
        Exception(std::string msg)  : msg_(msg) {}
        virtual ~Exception() throw() {}
        virtual char const* what() const throw() { return msg_.c_str(); }
    };

    /// An assert that throws an exception instead of aborting the program.
#ifndef NDEBUG
#define gassert(cond, msg)                              \
    if (!(cond)) {                                      \
        std::ostringstream message;                     \
        message << __FILE__ << ":"                      \
                << std::setw(4) << __LINE__ << ": "     \
                << msg << std::endl;                    \
        throw Exception(message.str());                 \
    }
#else
#define gassert(cond, msg) (void) (0)
#endif

    /// This functions takes a datum and returns a string representation.
    template<typename T>
    inline std::string toString(const T& object) {
        std::stringstream result;
        result << object;
        return result.str();
    }

    /// Allows printing of all boost::python wrapped python objects.
    inline std::ostream& operator<<(std::ostream& os, const python::object& object) {
        os << python::extract<std::string>(object.attr("__str__")())();
        return os;
    }
}
#endif
