#pragma once
#include <cstdint>
#include <vector_types.h>
#include <crt/host_defines.h>

#include "common.h"
#include "config.h"

Result initializeRadixSort(SimulationConfig const& config);
Result radixSort(uint32_t* inputs, uint32_t** p_out_idxs, size_t n);

Result __applyOrder(float3* d_out_values, uint32_t* d_indexes, size_t count);
