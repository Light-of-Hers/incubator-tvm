//
// Created by herlight on 2020/3/30.
//

#ifndef TVM_SRC_CONTRIB_BANG_BANG_COMMON_H_
#define TVM_SRC_CONTRIB_BANG_BANG_COMMON_H_

#include <cnrt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <dmlc/logging.h>
#include <cnrt.h>
#include "../../runtime/workspace_pool.h"

namespace tvm {
namespace runtime {

class GeneralChecker {
public:
  GeneralChecker(const char *file, int line) : file(file), line(line) {}
  friend cnrtRet_t operator+(cnrtRet_t ret, const GeneralChecker &ck) {
    if (ret != CNRT_RET_SUCCESS) {
      LOG(FATAL)
          << ck.file << ":" << ck.line << ": cnrt-call failed: " << cnrtGetErrorStr(ret);
    }
    return ret;
  }
  friend int operator+(int ret, const GeneralChecker &ck) {
    if (ret < 0) {
      LOG(FATAL)
          << ck.file << ":" << ck.line << ": system-call failed: " << strerror(errno);
    }
    return ret;
  }
  friend void *operator+(void *ret, const GeneralChecker &ck) {
    if (ret == nullptr) {
      LOG(FATAL)
          << ck.file << ":" << ck.line << ": system-call failed: " << strerror(errno);
    }
    return ret;
  }

private:
  const char *file;
  int line;
};
class DeferGuard {
public:
  template<typename F>
  DeferGuard(F func) : func(func) {}

  ~DeferGuard() { func(); }

private:
  std::function<void(void)> func;
};
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define CK +GeneralChecker(__FILE__, __LINE__)
#define defer DeferGuard CAT(__defer_guard_, __LINE__) = [&]() -> void

void *bangLoadModuleFromFile(const char *path);
void *bangLoadModuleFromMem(const char *src);
void *bangExtractSymbolFromModule(void *mod, const char *sym_name);
void bangUnloadModule(void *mod);

class BANGThreadEntry {
public:
  BANGThreadEntry();
  cnrtQueue_t queue{nullptr};
  WorkspacePool pool;
  static BANGThreadEntry *ThreadLocal();
};

}
}

#endif //TVM_SRC_CONTRIB_BANG_BANG_COMMON_H
