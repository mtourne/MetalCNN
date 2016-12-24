#include <metal_stdlib>

using namespace metal;

kernel void sigmoid(device float *x [[ buffer(0) ]],
                    device float *y [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]])
{
  y[id] = 1.0 / (1.0 + exp(-x[id]));
}
