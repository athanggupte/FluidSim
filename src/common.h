#pragma once
#include <cstdio>

#define XSTR(str) #str
#define STR(str) XSTR(str)

#define logDebug(fmt, ...) fprintf(stdout, "[Debug] (" __FILE__ ":" STR(__LINE__) ") :: " fmt "\n", ##__VA_ARGS__)
#define logError(fmt, ...) fprintf(stderr, "[Error] (" __FILE__ ":" STR(__LINE__) ") :: " fmt "\n", ##__VA_ARGS__)