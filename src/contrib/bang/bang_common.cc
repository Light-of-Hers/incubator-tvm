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

void *bangLoadModuleFromMem(const char *src) {
  CHECK(system(nullptr) != 0)
    << "Cannot use shell";

  char tmp_src[] = "/tmp/bang_srcXXXXXX";
  char tmp_obj[] = "/tmp/bang_objXXXXXX";
  char tmp_lib[] = "/tmp/bang_libXXXXXX";

  int fd = mkstemp(tmp_src);
  mkstemp(tmp_obj), mkstemp(tmp_lib);

  write(fd, src, strlen(src));
  close(fd);

  char cmd_buf[256];
  sprintf(cmd_buf, "cncc -x bang --bang-mlu-arch=MLU270 -O2 -c %s -o %s && "
                   "clang++ -shared %s -o %s",
          tmp_src, tmp_obj, tmp_obj, tmp_lib);
  system(cmd_buf);

  void *lib = dlopen(tmp_lib, RTLD_NOW);
  unlink(tmp_src), unlink(tmp_obj), unlink(tmp_lib);
  CHECK(lib != nullptr);
  return lib;
}
void *bangExtractSymbolFromModule(void *mod, const char *sym_name) {
  return dlsym(mod, sym_name) CK;
}
void *bangLoadModuleFromFile(const char *path) {
  try {
    char tmp_obj[] = "/tmp/bang_objXXXXXX", tmp_lib[] = "/tmp/bang_libXXXXXX";
    int fd = mkstemp(tmp_obj) CK;
    defer { unlink(tmp_obj); };
    close(fd);
    fd = mkstemp(tmp_lib) CK;
    defer { unlink(tmp_lib); };
    close(fd);

    pid_t pid = fork() CK;
    if (pid == 0) {
      execlp("cncc", "cncc", "-x", "bang", "--bang-mlu-arch=MLU270",
             "-O2", "-c", path, "-o", tmp_obj, nullptr) CK;
      return nullptr;
    }
    wait(nullptr) CK;

    pid = fork() CK;
    if (pid == 0) {
      execlp("clang++", "clang++", "-shared", tmp_obj, "-o", tmp_lib, nullptr) CK;
      return nullptr;
    }
    wait(nullptr) CK;

    return dlopen(tmp_lib, RTLD_NOW) CK;
  } catch (const std::runtime_error &err) {
    throw err;
  }
}
void bangUnloadModule(void *mod) {
  dlclose(mod);
}

}
}