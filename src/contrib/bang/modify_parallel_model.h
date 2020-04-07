//
// Created by herlight on 2020/3/28.
//

#ifndef TVM_SRC_CONTRIB_BANG_CHANGE_PARALLEL_MODEL_H_
#define TVM_SRC_CONTRIB_BANG_CHANGE_PARALLEL_MODEL_H_

#include <tvm/tir/stmt.h>

namespace tvm {
namespace tir {

Stmt ModifyParallelModel(const Stmt& stmt);

namespace attr {
constexpr const char *thread_loop = "thread_loop";
}

}
}

#endif //TVM_SRC_CONTRIB_BANG_CHANGE_PARALLEL_MODEL_H
