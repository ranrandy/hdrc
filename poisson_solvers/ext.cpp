#include <torch/extension.h>
#include "pycall.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve", &solve);
}
