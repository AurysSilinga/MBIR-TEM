#define NO_IMPORT_ARRAY

#include "gloripex.hpp"
#include "array.hpp"

namespace gloripex {
    using namespace boost::python;
    /// Converts a numpy array to an Array instance if possible. Uses
    /// constructor of Array class, but needs to manually generate a boost
    /// object from raw pointer..
    template <class T>
    struct ArrayConverter {
        ArrayConverter() {
            converter::registry::push_back(&convertible, &construct, type_id<T>());
        }

        static void* convertible(PyObject* obj_ptr) {
           return is_object_convertible_to_array<T>(obj_ptr) ? obj_ptr : 0;
        }

      /// See here for an explanation
      /// http://mail.python.org/pipermail/cplusplus-sig/2008-October/013895.html
        static void construct(PyObject* obj_ptr,
                              converter::rvalue_from_python_stage1_data* data)
        {
            object obj(handle<>(borrowed(obj_ptr)));
            data->convertible = (reinterpret_cast<converter::rvalue_from_python_storage<T>*> (data))->storage.bytes;
            new (data->convertible) T(obj);
        }
    };

  // Add converters as necessary
  ArrayConverter<Array<cub_t, 1> > array_cub_t_1_converter;
  ArrayConverter<Array<cub_t, 2> > array_cub_t_2_converter;
  ArrayConverter<Array<cub_t, 3> > array_cub_t_3_converter;

  ArrayConverter<Array<float, 1> > array_float_1_converter;
  ArrayConverter<Array<float, 2> > array_float_2_converter;
  ArrayConverter<Array<float, 3> > array_float_3_converter;

  ArrayConverter<Array<double, 1> > array_double_1_converter;
  ArrayConverter<Array<double, 2> > array_double_2_converter;
  ArrayConverter<Array<double, 3> > array_double_3_converter;

  ArrayConverter<Array<size_t, 1> > array_sizet_1_converter;
}
