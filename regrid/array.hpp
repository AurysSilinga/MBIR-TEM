//
// Copyright 2013 by Forschungszentrum Juelich GmbH
//

#ifndef GLORIPEX_ARRAY_HPP
#define GLORIPEX_ARRAY_HPP

#include "numpy.hpp"

namespace gloripex {

#pragma GCC diagnostic ignored "-Wold-style-cast"
    /// Determines whether a python object is a numpy array fitting to the Array
    /// type supplied by T.
    ///
    /// Example:
    /// if (is_object_convertible_to_array<Array<double, 3> >(obj_ptr)) { ... }
    template<typename T>
    bool is_object_convertible_to_array(PyObject* obj_ptr) {
        return (PyArray_Check(obj_ptr) &&
                PyArray_TYPE(obj_ptr) == numpy::getTypeID(typename T::value_type()) &&
                PyArray_NDIM(obj_ptr) == T::ndim());
    }
#pragma GCC diagnostic warning "-Wold-style-cast"

    /// \brief A fast array class suited to our needs. Based on the Blitz++
    /// concept without the fancy stuff.
    template<typename T, int ndimm>
    struct Array {
        // some typedefs for making this STL compatible
        typedef Array<T, ndimm> type;
        typedef T value_type;
        typedef T* iterator;
        typedef const T* const_iterator;
        typedef T& reference;
        typedef const T& const_reference;
        typedef size_t size_type;

        /// Function that provides a neat wrapper around numpy arrays. No copying is
        /// performed. User needs to supply dimensionality and type, but it is
        /// checked during runtime, if it fits!
        ///
        /// object is const, but this class does not really care about this...
        /// Instantiate a const Array if object should really be const.
        Array(const python::object& object) :
            object_(object),
            data_(static_cast<T*>(numpy::getArrayData(this->object_)))
        {
            this->initialise();
        }

        /// This function creates a numpy array. It may be accessed later on
        /// with the object member function, e.g. for returning it to python.
        Array(const std::vector<npy_intp>& dims) :
            object_(numpy::createFromScratch<T>(dims)),
            data_(static_cast<T*>(numpy::getArrayData(this->object_)))
        {
            this->initialise();
        }

        Array(const Array<T, ndimm>& other) :
            object_(other.object_),
            data_(static_cast<T*>(numpy::getArrayData(this->object_)))
        {
            this->initialise();
        }

        /// 1D access
        inline const T& operator() (size_type idx0) const {
            static_assert(ndimm == 1, "dimensionality does not fit function call");
            return this->data()[idx0 * this->stride(0)];
        }
        /// 1D access
        inline T& operator() (size_type idx0) {
            static_assert(ndimm == 1, "dimensionality does not fit function call");
            return this->data()[idx0 * this->stride(0)];
        }
        /// 2D access
        inline const T& operator() (size_type idx0, size_type idx1) const {
            static_assert(ndimm == 2, "dimensionality does not fit function call");
            return this->data()[idx0 * this->stride(0) + idx1 * this->stride(1)];
        }
        /// 2D access
        inline T& operator() (size_type idx0, size_type idx1) {
            static_assert(ndimm == 2, "dimensionality does not fit function call");
            return this->data()[idx0 * this->stride(0) + idx1 * this->stride(1)];
        }
        /// 3D access
        inline const T& operator() (size_type idx0, size_type idx1, size_type idx2) const {
            static_assert(ndimm == 3, "dimensionality does not fit function call");
            return this->data()[idx0 * this->stride(0) + idx1 * this->stride(1) +
                                idx2 * this->stride(2)];
        }
        /// 3D access
        inline T& operator() (size_type idx0, size_type idx1, size_type idx2) {
            static_assert(ndimm == 3, "dimensionality does not fit function call");
            return this->data()[idx0 * this->stride(0) + idx1 * this->stride(1) +
                                idx2 * this->stride(2)];
        }

        inline iterator begin() { return this->data(); }
        inline const_iterator begin() const { return this->data(); }
        inline iterator end() { return this->data() + this->size(); }
        inline const_iterator end()  const { return this->data() + this->size(); }

        /// returns the number of dimensions.
        static inline int ndim() { return ndimm; }

        /// returns the total size of the array
        size_type size() const {
            size_type result = shape(0);
            for (int i = 1; i < this->ndim(); ++ i) {
                result *= this->shape(i);
            }
            return result;
        }
        /// This function explores the shape of the array.
        inline size_type shape(size_type dim) const {
            assert(0 <= dim && dim < ndimm);
            return static_cast<size_type>(numpy::getArrayDims(this->object_)[dim]);
        }
        /// This function explores the strides of the array.
        inline size_type stride(size_type dim) const { return this->stride_[dim]; }
        /// Returns pointer to underlying data.
        inline T* data() const { return this->data_; }

        /// Returns the wrapped object.
        inline python::object object() const { return this->object_; }

        template<typename T2, int ndim2>
        friend std::ostream& operator<<(std::ostream& os, const Array<T2, ndim2>& a);

    private:

        void initialise() {
            assert(is_object_convertible_to_array<type>(this->object_.ptr()));

            for (int i = 0; i < ndimm; ++ i) {
                this->stride_[i] = numpy::getArrayStrides(this->object_)[i] / sizeof(T);
            }
        }

    protected:
        /// Reference to python array object
        python::object object_;
        /// Holds the a pointer to the memory of the array (regardless of who manages it).
        T* data_;
        /// \brief Stores the stride of the array (e.g., 20(=4*5) and 5 for a
        /// 3x4x5 array).
        size_type stride_[ndimm];
    };

    // ****************************************************************************

    template<typename T2, int ndim2>
    std::ostream& operator<<(std::ostream& os, const Array<T2, ndim2>& a) {
        os << python::extract<std::string>(a.object_.attr("__str__")())();
        return os;
    }
}

#endif
