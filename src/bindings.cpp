#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace py = pybind11;

py::tuple discretize_column(py::array_t<float> input, float sharpness) {
    auto buf = input.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    if (size == 0) {
        return py::make_tuple(py::array_t<int>(), 0.0f, 0.0f, 0.0f);
    }

    // 1. Находим min/max (игнорируя NaN)
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < size; i++) {
        if (!std::isnan(ptr[i])) {
            if (ptr[i] < min_val) min_val = ptr[i];
            if (ptr[i] > max_val) max_val = ptr[i];
        }
    }

    // 2. Если все значения NaN
    if (min_val == std::numeric_limits<float>::max()) {
        std::vector<int> result(size, -1);
        py::array_t<int> binned_array({size}, {sizeof(int)}, result.data());
        return py::make_tuple(binned_array, 0.0f, 0.0f, 0.0f);
    }

    // 3. Вычисляем интервалы
    int n_intervals = static_cast<int>(std::round(2.0f / sharpness));
    if (n_intervals < 1) n_intervals = 1;
    
    float step = (max_val - min_val) / n_intervals;
    if (step < 1e-10f) step = 1.0f;

    // 4. Дискретизация
    std::vector<int> result(size);
    for (size_t i = 0; i < size; i++) {
        if (std::isnan(ptr[i])) {
            result[i] = -1;
        } else {
            int idx = static_cast<int>((ptr[i] - min_val) / step);
            idx = std::max(0, std::min(idx, n_intervals - 1));
            result[i] = idx;
        }
    }

    py::array_t<int> binned_array({size}, {sizeof(int)}, result.data());
    return py::make_tuple(binned_array, min_val, max_val, step);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for ih-prep";
    m.def("discretize_column", &discretize_column,
          "Discretize a single column with given sharpness, NaN encoded as -1");
}
