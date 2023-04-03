#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "indexremove/indexremove.h"

namespace py = pybind11;

auto remove_(py::array_t<long> index, py::array_t<long> removed) {
  auto index_ = index.request();
  auto removed_ = removed.request();
  remove(
    index_.shape[0],
    static_cast<long*>(index_.ptr),
    removed_.shape[0],
    static_cast<long*>(removed_.ptr)
  );
}

PYBIND11_MODULE(indexremove, m) {
    m.doc() = "indexremove extension";

    m.def("remove", remove_);
}
