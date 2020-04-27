//
// Created by herlight on 2020/3/24.
//

#include "bang_module.h"
#include "bang_common.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../runtime/pack_args.h"
#include "../../target/build_common.h"
#include "../../runtime/file_util.h"
#include "./codegen_bang.h"
#include <string>
#include <array>
#include <mutex>
#include <utility>
#include <tvm/ir/module.h>
#include <unordered_map>

namespace tvm {
namespace runtime {

class BANGModuleNode final : public ModuleNode {
public:
  BANGModuleNode(std::string src, std::unordered_map<std::string, FunctionInfo> fmap)
      : src_(std::move(src)), fmap_(std::move(fmap)) {}

  ~BANGModuleNode() override {
    for (auto dm : mod_) {
      bangUnloadModule(dm.second);
    }
  }
  const char *type_key() const override {
    return "bang";
  }
  PackedFunc GetFunction(const std::string &name, const ObjectPtr<Object> &sptr_to_self) override;
  std::string GetSource(const std::string &format) override {
    return src_;
  }
  void SaveToFile(const std::string &file_name, const std::string &format) override {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "mlu") {
      CHECK_NE(src_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, src_);
    } else {
      LOG(FATAL) << "Cannot support format other than mlu";
    }
  }
  void SaveToBinary(dmlc::Stream *stream) override {
    stream->Write(fmap_);
    stream->Write(src_);
  }
  void *GetSymbol(cnrtDev_t dev, const std::string &sym_name) {
    try {
      mtx_.lock();
      defer { mtx_.unlock(); };
      if (mod_.count(dev) == 0)
        mod_[dev] = bangLoadModuleFromMem(src_.c_str(), src_.size());
      return bangExtractSymbolFromModule(mod_[dev], sym_name.c_str());
    } catch (const std::exception &err) {
      throw err;
    }
  }

private:
  std::string src_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::map<cnrtDev_t, void *> mod_;
  std::mutex mtx_;
};
class BANGWrappedFunc {
public:
  void Init(BANGModuleNode *m,
            ObjectPtr<Object> sptr,
            const std::string &func_name,
            std::vector<DLDataType> arg_types,
            const std::vector<std::string> &thread_axis_tags) {
    m_ = m;
    sptr_ = std::move(sptr);
    func_name_ = func_name;
    arg_types_ = std::move(arg_types);
    thread_axis_cfg_.Init(arg_types_.size(), thread_axis_tags);
  }
  void operator()(TVMArgs args, TVMRetValue *rv) const {
    cnrtDev_t dev;
    cnrtGetCurrentDevice(&dev) CK;
    if (fcache_.count(dev) == 0) {
      fcache_[dev] = m_->GetSymbol(dev, func_name_);
    }
    std::vector<ArgUnion> holders(args.size());
    cnrtKernelParamsBuffer_t pms;
    cnrtGetKernelParamsBuffer(&pms) CK;
    for (int i = 0; i < arg_types_.size(); ++i) {
      CHECK_EQ(arg_types_[i].lanes, 1U);
      auto code = arg_types_[i].code;
      auto bits = arg_types_[i].bits;
      CHECK(bits == 32U || bits == 64U);
      if (code == kDLFloat) {
        holders[i].v_float32 = static_cast<float>(args.values[i].v_float64);
        cnrtKernelParamsBufferAddParam(pms, &holders[i], sizeof(float));
      } else if (code == kDLInt) {
        holders[i].v_int32 = static_cast<int32_t>(args.values[i].v_int64);
        cnrtKernelParamsBufferAddParam(pms, &holders[i], sizeof(int32_t));
      } else if (code == kDLUInt) {
        holders[i].v_uint32 = static_cast<uint32_t>(args.values[i].v_int64);
        cnrtKernelParamsBufferAddParam(pms, &holders[i], sizeof(uint32_t));
      } else if (code == kTVMOpaqueHandle) {
        cnrtKernelParamsBufferAddParam(pms, (void *) &args.values[i].v_handle, sizeof(void *));
      }
    }
    auto que = BANGThreadEntry::ThreadLocal()->queue;
    auto wl = thread_axis_cfg_.Extract(args);
    uint32_t taskDimX = wl.grid_dim(0) * wl.grid_dim(1) * wl.grid_dim(2);
    taskDimX = (taskDimX + 15) / 16 * 16;
    cnrtDim3_t dim{taskDimX, 1, 1};
    auto fty = CNRT_FUNC_TYPE_UNION4;
    cnrtInvokeKernel_V2(fcache_[dev], dim, pms, fty, que) CK;
  }

private:
  BANGModuleNode *m_{nullptr};
  ObjectPtr<Object> sptr_;
  std::vector<DLDataType> arg_types_;
  std::string func_name_;
  mutable std::map<cnrtDev_t, void *> fcache_;
  ThreadAxisConfig thread_axis_cfg_;
};

PackedFunc BANGModuleNode::GetFunction(
    const std::string &name,
    const ObjectPtr<Object> &sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
    << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end())
    return PackedFunc();
  const auto &info = it->second;
  BANGWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types, info.thread_axis_tags);
  return PackedFunc(f);
}

Module BANGModuleCreate(std::string src,
                        std::unordered_map<std::string, FunctionInfo> fmap) {
  auto n = make_object<BANGModuleNode>(std::move(src), std::move(fmap));
  return Module(n);
}

Module BANGModuleLoadFile(const std::string &file_name,
                          const std::string &format) {
  std::string src;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &src);
  LoadMetaDataFromFile(meta_file, &fmap);
  return BANGModuleCreate(std::move(src), std::move(fmap));
}

Module BANGModuleLoadBinary(void *strm) {
  auto stream = static_cast<dmlc::Stream *>(strm);
  std::string src;
  std::unordered_map<std::string, FunctionInfo> fmap;
  stream->Read(&fmap);
  stream->Read(&src);
  return BANGModuleCreate(std::move(src), std::move(fmap));
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_mlu")
.set_body_typed(BANGModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_bang")
.set_body_typed(BANGModuleLoadBinary);

}

namespace codegen {

runtime::Module BuildBANG(IRModule mod) {
  bool output_ssa = false;

  CodeGenBANG cgb;
  cgb.Init(output_ssa);

  for (auto kv : mod->functions) {
    CHECK(kv.second->IsInstance<PrimFuncNode>())
      << "CodeGenCUDA: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    CHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
      << "CodeGenCUDA: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cgb.AddFunction(f);
  }

  return runtime::BANGModuleCreate(cgb.Finish(), ExtractFuncInfo(mod));
}

TVM_REGISTER_GLOBAL("target.build.ext_dev")
.set_body_typed(BuildBANG);

}

}