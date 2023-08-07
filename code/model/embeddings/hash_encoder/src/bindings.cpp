#include <torch/extension.h>

#include "hashencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash_encode_forward", &hash_encode_fwd, "hash encode forward (CUDA)");
    m.def("hash_encode_backward", &hash_encode_bwd, "hash encode backward (CUDA)");
}