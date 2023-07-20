#ifndef __HASH_ENCODER_H
#define __HASH_ENCODER_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>
/*
-- hash_encode_fwd --
  Args:
  - Inputs: [B, D] - float , in [0,1]
  - embeddings: [s0, C] -float 
  - offsets: [L+1] -uint32_t  (stdint)
  - outputs: [B, L*C] - float 
  - H: base resolution  - uint32_t
*/
void hash_encode_fwd(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx);
void hash_encode_bwd(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs);

#endif