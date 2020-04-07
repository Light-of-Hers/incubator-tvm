//
// Created by herlight on 2020/3/24.
//

#ifndef TVM_SRC_CONTRIB_BANG_BANG_MODULE_H_
#define TVM_SRC_CONTRIB_BANG_BANG_MODULE_H_

#include <tvm/runtime/module.h>
#include <map>
#include "../../runtime/meta_data.h"

namespace tvm {
namespace runtime {

Module BANGModuleCreate(
    std::string src,
    std::map<std::string, FunctionInfo> fmap);

}
}

#endif //TVM_SRC_CONTRIB_BANG_BANG_MODULE_H
