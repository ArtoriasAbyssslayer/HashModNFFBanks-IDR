import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(
    name='_hash_encoder',
    sources=[os.path.join(_src_path, 'src', f) for f in [
        'hashencoder.cu',
        'bindings.cpp',
    ]],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['--use_fast_math'],
    extra_ldflags=['-O3'],
    verbose=True,  # Set verbose to True to enable detailed output.
)

__all__ = ['_backend']
