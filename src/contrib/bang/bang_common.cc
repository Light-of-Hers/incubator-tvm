//
// Created by herlight on 2020/3/30.
//

#include "bang_common.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <dlfcn.h>

namespace tvm {
namespace runtime {

void *bangLoadModuleFromMem(const char *src, size_t len) {
  char tmp_src[] = "/tmp/bang_srcXXXXXX";
  char tmp_obj[] = "/tmp/bang_objXXXXXX";
  char tmp_lib[] = "/tmp/bang_libXXXXXX";
  try {
    CHECK(system(nullptr) != 0)
      << "Cannot use shell";

    int fd = mkstemp(tmp_src) CK;
    close(mkstemp(tmp_obj) CK), close(mkstemp(tmp_lib) CK);

    write(fd, src, len);
    close(fd);

    char cmd[256];
    sprintf(cmd, "cncc -x bang --bang-mlu-arch=MLU270 --bang-device-only -std=c++1z -O3 -c %s -o %s && "
                 "clang++ -shared %s -o %s",
            tmp_src, tmp_obj, tmp_obj, tmp_lib);
    system(cmd) CK;
    unlink(tmp_src), unlink(tmp_obj);

    void *lib = dlopen(tmp_lib, RTLD_NOW) CK;
    unlink(tmp_lib);
    return lib;
  } catch (const std::exception &err) {
    unlink(tmp_src), unlink(tmp_obj), unlink(tmp_lib);
    throw err;
  }
}
void *bangExtractSymbolFromModule(void *mod, const char *sym_name) {
  return dlsym(mod, sym_name) CK;
}
void bangUnloadModule(void *mod) {
  dlclose(mod);
}

}
}