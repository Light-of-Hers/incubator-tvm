//
// Created by herlight on 2020/3/30.
//

#include "bang_common.h"
#include <cnrt.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/var.h>
#include <sstream>
#include <cstring>
#include <dmlc/thread_local.h>

namespace tvm {
namespace runtime {

class BANGDeviceAPI final : public DeviceAPI {
public:
  void SetDevice(TVMContext ctx) override {
    cnrtInit(0) CK;
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, ctx.device_id) CK;
    cnrtSetCurrentDevice(dev) CK;
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue *rv) override {
    switch (kind) {
    case kExist: {
      cnrtDeviceInfo_t info;
      *rv = cnrtGetDeviceInfo(&info, ctx.device_id) == CNRT_RET_SUCCESS;
      return;
    }
    case kComputeVersion: {
      static const char *versions[] = {
          "1H8", "1H16", "1H8MINI", "MLU100", "MLU270", "MLU220"
      };
      cnrtDeviceInfo_t info;
      cnrtGetDeviceInfo(&info, ctx.device_id) CK;
      *rv = std::string(versions[info.core_version]);
      return;
    }
    case kMultiProcessorCount : {
      cnrtDeviceInfo_t info;
      cnrtGetDeviceInfo(&info, ctx.device_id) CK;
      *rv = info.core_num;
      return;
    }
    case kDeviceName: {
      cnrtDeviceInfo_t info;
      cnrtGetDeviceInfo(&info, ctx.device_id) CK;
      *rv = std::string(info.device_name);
      return;
    }
    default: {
      *rv = 0;
      return;
    }
    }
  }
  void *AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       DLDataType type_hint) override {
    CHECK_EQ(alignment % 32, 0U)
      << "MLU space is aligned at 32 bytes";
    void *ret{nullptr};
    if (ctx.device_type == kDLCPUPinned) {
      cnrtMallocHost(&ret, nbytes, CNRT_MEMTYPE_LOCKED) CK;
    } else {
      SetDevice(ctx);
      cnrtMalloc(&ret, nbytes) CK;
    }
    return ret;
  }
  void FreeDataSpace(TVMContext ctx, void *ptr) override {
    if (ctx.device_type == kDLCPUPinned) {
      cnrtFreeHost(ptr) CK;
    } else {
      SetDevice(ctx);
      cnrtFree(ptr) CK;
    }
  }
  void CopyDataFromTo(const void *from,
                      size_t from_offset,
                      void *to,
                      size_t to_offset,
                      size_t num_bytes,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      DLDataType type_hint,
                      TVMStreamHandle stream) override {
    auto que = static_cast<cnrtQueue_t>(stream);
    from = static_cast<const char *>(from) + from_offset;
    to = static_cast<char *>(to) + to_offset;

    if (ctx_from.device_type == kDLCPUPinned) {
      ctx_from.device_type = kDLCPU;
    }
    if (ctx_to.device_type == kDLCPUPinned) {
      ctx_to.device_type = kDLCPU;
    }

    if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLCPU) {
      memcpy(to, from, num_bytes);
    } else if (ctx_from.device_type == kDLExtDev && ctx_to.device_type == kDLExtDev) {
      SetDevice(ctx_from);
      if (ctx_from.device_id == ctx_to.device_id) {
        MLUCopy(from, to, num_bytes, CNRT_MEM_TRANS_DIR_DEV2DEV, que);
      } else {
        cnrtMemcpyPeer(to, ctx_to.device_id, const_cast<void *>(from), ctx_from.device_id, num_bytes) CK;
      }
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLExtDev) {
      SetDevice(ctx_to);
      MLUCopy(from, to, num_bytes, CNRT_MEM_TRANS_DIR_HOST2DEV, que);
    } else if (ctx_from.device_type == kDLExtDev && ctx_to.device_type == kDLCPU) {
      SetDevice(ctx_from);
      MLUCopy(from, to, num_bytes, CNRT_MEM_TRANS_DIR_DEV2HOST, que);
    } else {
      LOG(FATAL)
          << "Cannot support data copy from "
          << DeviceName(ctx_from.device_type) << " to "
          << DeviceName(ctx_to.device_type);
    }
  }
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) override {
    SetDevice(ctx);
    auto que = static_cast<cnrtQueue_t>(stream);
    cnrtSyncQueue(que) CK;
  }
  TVMStreamHandle CreateStream(TVMContext ctx) override {
    SetDevice(ctx);
    cnrtQueue_t que;
    cnrtCreateQueue(&que) CK;
    return static_cast<TVMStreamHandle>(que);
  }
  void FreeStream(TVMContext ctx, TVMStreamHandle stream) override {
    SetDevice(ctx);
    auto que = static_cast<cnrtQueue_t>(stream);
    cnrtDestroyQueue(que) CK;
  }
  void SyncStreamFromTo(TVMContext ctx,
                        TVMStreamHandle event_src,
                        TVMStreamHandle event_dst) override {
    SetDevice(ctx);
    auto src_que = static_cast<cnrtQueue_t>(event_src);
    auto dst_que = static_cast<cnrtQueue_t>(event_dst);
    cnrtNotifier_t evt;
    cnrtCreateNotifier(&evt) CK;
    cnrtPlaceNotifier(evt, src_que) CK;
    cnrtQueueWaitNotifier(evt, dst_que, 0) CK;
    cnrtDestroyNotifier(&evt) CK;
  }
  void SetStream(TVMContext ctx, TVMStreamHandle stream) override {
    BANGThreadEntry::ThreadLocal()->queue = static_cast<cnrtQueue_t>(stream);
  }
  void *AllocWorkspace(TVMContext ctx, size_t nbytes, DLDataType type_hint) override {
    return BANGThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, nbytes);
  }
  void FreeWorkspace(TVMContext ctx, void *ptr) override {
    BANGThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, ptr);
  }
  static std::shared_ptr<BANGDeviceAPI> &Global() {
    static auto singleton = std::make_shared<BANGDeviceAPI>();
    return singleton;
  }
private:
  static void MLUCopy(const void *from,
                      void *to,
                      size_t size,
                      cnrtMemTransDir_t dir,
                      cnrtQueue_t que) {
    if (que) {
      cnrtMemcpyAsync(to, const_cast<void *>(from), size, que, dir) CK;
    } else {
      cnrtMemcpy(to, const_cast<void *>(from), size, dir) CK;
    }
  }
};

using BANGThreadStore = dmlc::ThreadLocalStore<BANGThreadEntry>;
BANGThreadEntry::BANGThreadEntry()
    : pool(kDLExtDev, BANGDeviceAPI::Global()) {}
BANGThreadEntry *BANGThreadEntry::ThreadLocal() {
  return BANGThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.ext_dev")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  DeviceAPI *ptr = BANGDeviceAPI::Global().get();
  *rv = static_cast<void *>(ptr);
});

TVM_REGISTER_GLOBAL("tvm.info.mem.local.nram")
.set_body_typed([]() {
  auto info = make_object<MemoryInfoNode>();
  info->unit_bits = 8;
  info->max_num_bits = 512 * 1024 * 8;
  info->max_simd_bits = 64 * 8;
  return MemoryInfo(info);
});

TVM_REGISTER_GLOBAL("tvm.info.mem.local.wram")
.set_body_typed([]() {
  auto info = make_object<MemoryInfoNode>();
  info->unit_bits = 8;
  info->max_num_bits = 1024 * 1024 * 8;
  info->max_simd_bits = 64 * 8;
  return MemoryInfo(info);
});

}
}