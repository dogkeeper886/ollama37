// K80 port: force-included (-include) at the top of every TU.
//
// libcu++ 11.4 <cuda/std/cmath> does `using ::isnan;`/`using ::isunordered;` etc.,
// but on glibc those are macros and libstdc++ <cmath> provides the functions only
// in std:: (no global :: function forms) — so the toolkit header fails with
// "X has not been declared in '::'". A header-shim can't catch it because the
// toolkit reaches cmath via internal relative includes. Force-including this
// prelude makes the global :: functions exist before any cmath is parsed.
#pragma once

#include <cmath>

// C99 comparison functions: no CUDA device :: builtins, so export unconditionally.
using std::isgreater;
using std::isgreaterequal;
using std::isless;
using std::islessequal;
using std::islessgreater;
using std::isunordered;

// Classification functions: CUDA provides device :: builtins in the device pass,
// so exporting std:: there makes direct isnan(x)/etc. ambiguous. Only export in
// the host pass (where libcu++'s `using ::isnan` would otherwise fail).
#if !defined(__CUDA_ARCH__)
using std::fpclassify;
using std::isfinite;
using std::isinf;
using std::isnan;
using std::isnormal;
using std::signbit;
#endif
