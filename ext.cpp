#include <torch/extension.h>
#include "hdrc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hdrcCUDA", &hdrcCUDA);
}
