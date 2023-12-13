#pragma once
#include <cstdint>
#include <crt/host_defines.h>

#include "common.h"
#include "config.h"

Result initializeRadixSort(SimulationConfig const& config);
Result radixSort(uint32_t* outputs, uint32_t* inputs, size_t n);
