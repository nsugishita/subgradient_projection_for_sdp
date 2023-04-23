#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "uniquelist/unique_list.h"
#include <iostream>

namespace py = pybind11;

using cls = uniquelist::unique_list<int>;

// TODO Make UniqueList pickable.

PYBIND11_MODULE(uniquelist, m) {
    m.doc() = "uniquelist extension";

    py::class_<cls>(m, "UniqueList")
   .def(py::init<>())
   .def("size", &cls::size, "Return the number of items in the list")
   .def("push_back",
     py::overload_cast<const int&>(&cls::push_back),
     "Add an item at the end of the list if its' new"
    )
   .def("erase_nonzero",
     [](cls &a, py::array_t<int> removed) {
      auto removed_ = removed.request();
      return a.erase_nonzero(removed_.shape[0], static_cast<int*>(removed_.ptr));
     },
     "Erase items at given positions"
   )
   .def("index",
     [](const cls &a, int x) {
       int i = 0;
       for (auto item : a) {
         if (item == x) {
           return i;
         }
         ++i;
       }
       return -1;
     },
     "Search a give item in the list and return its index"
   )
   .def("display",
     [](const cls &a) {
       for (auto item : a) {
         std::cout << item << " ";
       }
       std::cout << std::endl;
     },
     "Print the items"
   )
   ;
}
