//
// Copyright 2013 by Forschungszentrum Juelich GmbH
//

#ifndef GLORIPEX_NUMPY
#define GLORIPEX_NUMPY


namespace gloripex {
    namespace numpy {
      typedef python::numeric::array ndarray;

        inline int getTypeID(float) { return NPY_FLOAT; }
        inline int getTypeID(double) { return NPY_DOUBLE; }
        inline int getTypeID(long double) { return NPY_LONGDOUBLE; }
        inline int getTypeID(signed char) { return NPY_BYTE; }
        inline int getTypeID(unsigned char) { return NPY_UBYTE; }
        inline int getTypeID(int) { return NPY_INT; }
        inline int getTypeID(unsigned int) { return NPY_UINT; }
        inline int getTypeID(short) { return NPY_SHORT; }
        inline int getTypeID(unsigned short) { return NPY_USHORT; }
        inline int getTypeID(unsigned long) { return NPY_ULONG; }
        inline int getTypeID(long) { return NPY_LONG; }


        inline int getArrayType(const python::object &x) {
            return reinterpret_cast<PyArrayObject*>(x.ptr())->descr->type_num;
        }

        inline npy_intp* getArrayDims(const python::object &x) {
            return reinterpret_cast<PyArrayObject*>(x.ptr())->dimensions;
        }

        inline npy_intp* getArrayStrides(const python::object &x) {
            return reinterpret_cast<PyArrayObject*>(x.ptr())->strides;
        }

        inline int getArrayNDims(const python::object &x) {
            return reinterpret_cast<PyArrayObject*>(x.ptr())->nd;
        }

        inline void* getArrayData(const python::object &x) {
            return reinterpret_cast<PyArrayObject*>(x.ptr())->data;
        }

        inline PyArray_Descr* getArrayDescription(const python::object &x) {
            return reinterpret_cast<PyArrayObject*>(x.ptr())->descr;
        }

        inline int getArrayItemsize(const python::object &x) {
            return reinterpret_cast<PyArrayObject*>(x.ptr())->descr->elsize;
        }

        template<typename T>
        inline T* getArrayDataAs(const python::object &x) {
            return reinterpret_cast<T*>(getArrayData(x));
        }

#pragma GCC diagnostic ignored "-Wold-style-cast"
        inline bool isArray(const python::object &x) {
            return PyArray_Check(x.ptr());
        }

        /// Creates an empty python array.
        template<typename T>
        inline ndarray createFromScratch(const std::vector<npy_intp>& dims) {
            PyObject* ptr = PyArray_SimpleNew(
                    static_cast<int>(dims.size()),
                    const_cast<npy_intp*>(&dims[0]),
                    getTypeID(T()));
            python::handle<> hndl(ptr);
            return ndarray(hndl);
        }

        /// Creates a python array from existing memory. As the lifetime of the
        /// python object cannot be controlled, it is rather dangerous to delete
        /// the memory. Use with utmost care.
        template<typename T>
        inline ndarray createFromData(
                const std::vector<npy_intp>& dims, T* data) {
            PyObject* ptr = PyArray_SimpleNewFromData(
                    static_cast<int>(dims.size()),
                    const_cast<npy_intp*>(&dims[0]),
                    getTypeID(T()),
                    data);
            python::handle<> hndl(ptr);
            return ndarray(hndl);
        }
#pragma GCC diagnostic warning "-Wold-style-cast"
    }
}

#endif
