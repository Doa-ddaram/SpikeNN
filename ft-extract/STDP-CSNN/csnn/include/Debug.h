#ifndef _DEBUG_H
#define _DEBUG_H

#include <stdexcept>

#ifdef DEBUG
#define ASSERT_DEBUG(EXPR) if(!(EXPR)) { throw std::runtime_error(std::string("Assert failed: ")+#EXPR); }
#else
#define ASSERT_DEBUG(EXPR)
#endif

#endif
