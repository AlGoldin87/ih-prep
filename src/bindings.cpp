#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

// Простая дискретизация (ядро из ih_lib)
py::tuple discretize_column(py::array_t<float> input, float sharpness) {
    auto buf = input.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    if (size == 0) {
        return py::make_tuple(py::array_t<int>(), 0.0f, 0.0f, 0.0f);
    }

    // Находим min и max
    float min_val = *std::min_element(ptr, ptr + size);
    float max_val = *std::max_element(ptr, ptr + size);

    int n_intervals = static_cast<int>(std::round(2.0f / sharpness));
    if (n_intervals < 1) n_intervals = 1;
    
    float step = (max_val - min_val) / n_intervals;
    if (step < 1e-10f) step = 1.0f;

    // Дискретизация
    std::vector<int> result(size);
    for (size_t i = 0; i < size; i++) {
        int idx = static_cast<int>((ptr[i] - min_val) / step);
        idx = std::max(0, std::min(idx, n_intervals - 1));
        result[i] = idx;
    }

    // Преобразуем в numpy массив
    py::array_t<int> binned_array({size}, {sizeof(int)}, result.data());

    return py::make_tuple(binned_array, min_val, max_val, step);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for ih-prep";
    m.def("discretize_column", &discretize_column,
          "Discretize a single column with given sharpness");
}
