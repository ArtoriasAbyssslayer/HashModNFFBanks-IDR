//#pragma once
#include <torch/extension.h>
#incude "hashencoder.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("hash_encode_fwd", &hash_encode_fwd, "hash encode forward pass (CUDA)");
    m.def("hash_encode_bwd", &hash_encode_bwd, "hash encode backward pass (CUDA)");
}