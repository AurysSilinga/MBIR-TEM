
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

namespace python = boost::python;

#include "numpy.hpp"
#include "array.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>

#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/natural_neighbor_coordinates_3.h>

struct Interpolate {

  typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
  typedef Kernel::FT CoordType;
  typedef Kernel::Point_3 Point;
  typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned, Kernel> Vb;
  typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
  typedef CGAL::Delaunay_triangulation_3<Kernel, Tds> DelaunayTriangulation;
  typedef DelaunayTriangulation::Vertex_handle VertexHandle;

  DelaunayTriangulation T_;
  const gloripex::Array<double, 2> data_;

  Interpolate(const gloripex::Array<double, 2>& data) :
    data_(data)
  {
    const auto size = data_.shape(0);
    for (size_t i = 0; i < size; ++ i) {
      this->T_.insert(Point(data_(i, 0), data_(i, 1), data_(i, 2)))->info() = i;
    }
  }

  // *****************************************************************************

  void interpolate(const gloripex::Array<double, 2>& coord2) const
  {
    gloripex::Array<double, 2>& coord = const_cast<gloripex::Array<double, 2>&>(coord2);
    // coordinate computation
    const auto size = coord.shape(0);

    for (size_t i = 0; i < size; ++ i) {
      Point p(coord(i, 0), coord(i, 1), coord(i, 2));
      std::vector<std::pair<VertexHandle, CoordType> > vertices;
      CoordType norm;

      CGAL::laplace_natural_neighbor_coordinates_3(this->T_, p, std::back_inserter(vertices), norm);

      if (! vertices.empty()) {
        for (int j = 3; j < data_.shape(1); ++ j) {
          coord(i, j) = 0;
        }
        for (const auto& vertex : vertices) {
          double weight = vertex.second / norm;
          for (int j = 3; j < data_.shape(1); ++ j) {
            coord(i, j) += weight * data_((vertex.first)->info(), j);
          }

        }
      } else {
        for (int j = 3; j < data_.shape(1); ++ j) {
          coord(i, j) = std::numeric_limits<double>::quiet_NaN();
        }
      }
    }
  }
};

BOOST_PYTHON_MODULE(regrid)
{
  python::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();

  python::class_<Interpolate>("regrid", python::init<const gloripex::Array<double, 2>&>())
    .def("__call__", &Interpolate::interpolate);
}

