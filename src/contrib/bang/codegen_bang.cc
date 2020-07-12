//
// Created by herlight on 2020/3/7.
//

#include "codegen_bang.h"
#include "./modify_parallel_model.h"
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/expr_functor.h>
#include "../../arith/pattern_match.h"
#include "./fmt/format.h"
#include "./fmt/ostream.h"

namespace tvm {
namespace codegen {

CodeGenBANG::CodeGenBANG() {
  restrict_keyword_ = "__restrict__";
}
std::string CodeGenBANG::Finish() {
  // header
  decl_stream
      << "#include \"mlu.h\"\n";

  // declare auxilary
  decl_stream
      << "#define ALIGN_DN(n, f) ((n) / (f) * (f))\n"
      << "#define ALIGN_UP(n, f) (((n) + (f) - 1) / (f) * (f))\n";

  // declare stream operations
  decl_stream
      << "#define DECL_MEMSET_OP(name, align) "
      << "template <typename T, int TotN, int VecN = ALIGN_DN(TotN, align)> "
      << "__mlu_func__ void name ## _ (T *ptr, T val) "
      << "{ if constexpr (VecN > 0) name (ptr, VecN, val); "
         "if constexpr (VecN < TotN) for (int i = VecN; i < TotN; ++i) ptr[i] = val; }\n";
  for (const auto &on : scope_map()) {
    fmt::print(decl_stream, "DECL_MEMSET_OP(__{}set, 64)\n", on.second);
  }
  decl_stream
      << "#define DECL_STRM_BIN_OP(name, op, align) "
      << "template <typename T, int TotN, int VecN = ALIGN_DN(TotN, align)> "
      << "__mlu_func__ void name ## _ (T *dst, T *src0, T *src1) "
      << "{ if constexpr (VecN > 0) name (dst, src0, src1, VecN); "
         "if constexpr (VecN < TotN) for (int i = VecN; i < TotN; ++i) dst[i] = src0[i] op src1[i]; }\n";
  for (const auto &on : bang_stream_binary_ops()) {
    fmt::print(decl_stream, "DECL_STRM_BIN_OP({}, {}, 64)\n",
               on.second, on.first);
  }
  decl_stream
      << "#define DECL_STRM_BIN_CONST_OP(name, op, align) "
      << "template <typename T, int TotN, int VecN = ALIGN_DN(TotN, align)> "
      << "__mlu_func__ void name ## _ (T *dst, T *src, T val) "
      << "{ if constexpr (VecN > 0) name (dst, src, val, VecN); "
         "if constexpr (VecN < TotN) for (int i = VecN; i < TotN; ++i) dst[i] = src[i] op val; }\n";
  for (const auto &on : bang_stream_binary_const_ops()) {
    fmt::print(decl_stream, "DECL_STRM_BIN_CONST_OP({}, {}, 64)\n",
               on.second, on.first);
  }
  decl_stream
      << "#define BLOCK_DIM_X " << par_var_dim_["blockIdx.x"] << '\n'
      << "#define BLOCK_DIM_Y " << par_var_dim_["blockIdx.y"] << '\n'
      << "#define BLOCK_DIM_Z " << par_var_dim_["blockIdx.z"] << '\n'
      << "#define THREAD_DIM_X " << par_var_dim_["threadIdx.x"] << "\n"
      << "#define THREAD_DIM_Y " << par_var_dim_["threadIdx.y"] << "\n"
      << "#define THREAD_DIM_Z " << par_var_dim_["threadIdx.z"] << "\n";
  decl_stream
      << "#define BLOCK_IDX_X (taskIdX / (BLOCK_DIM_Y * BLOCK_DIM_Z))\n"
      << "#define BLOCK_IDX_Y ((taskIdX / BLOCK_DIM_Z) % BLOCK_DIM_Y)\n"
      << "#define BLOCK_IDX_Z (taskIdX % BLOCK_DIM_Z)\n";
  decl_stream
      << "#define N_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)\n"
      << "#define N_THREAD (THREAD_DIM_X * THREAD_DIM_Y * THREAD_DIM_Z)\n";

  // declare thread-loop
  decl_stream
      << "#define THREAD_LOOP ";
  for (auto axis : {"threadIdx.x", "threadIdx.y", "threadIdx.z"}) {
    if (used_par_var_.count(axis) > 0) {
      auto var = par_var_map().at(axis);
      auto ext = par_var_dim_.at(axis);
      decl_stream
          << " for (int " << var << " = 0; " << var << " < " << ext << "; " << "++" << var << ")";
    }
  }
  decl_stream << '\n';

  // declare thread-idx mapping
  decl_stream
      << "#define THREAD_IDX ("
      << (used_par_var_.count("threadIdx.x") ? "threadIdx_x" : "0") << " * (THREAD_DIM_Y * THREAD_DIM_Z) + "
      << (used_par_var_.count("threadIdx.y") ? "threadIdx_y" : "0") << " * THREAD_DIM_Z + "
      << (used_par_var_.count("threadIdx.z") ? "threadIdx_z" : "0") << ")\n";

  // declare function pre-delcaration
  decl_stream
      << "#define PRE_DECL "
      << "if (taskIdX >= N_BLOCK) return; "
      << fmt::format("__nram__ uint8_t {}[{}]; ", nram_tmp_buf_.name(), nram_tmp_buf_.max_size())
      << (used_par_var_.count("blockIdx.x") ? "int blockIdx_x = BLOCK_IDX_X; " : "")
      << (used_par_var_.count("blockIdx.y") ? "int blockIdx_y = BLOCK_IDX_Y; " : "")
      << (used_par_var_.count("blockIdx.z") ? "int blockIdx_z = BLOCK_IDX_Z; " : "")
      << '\n';

  return CodeGenC::Finish();
}
void CodeGenBANG::PrintStorageSync(const CallNode *op) {
}
void CodeGenBANG::PrintStorageScope(const std::string &scope, std::ostream &os) {
}
void CodeGenBANG::PrintVecBinaryOp(const std::string &op, DataType t,
                                   PrimExpr lhs, PrimExpr rhs, std::ostream &os) {
  auto is_src_expr_root = is_src_expr_ && src_expr_depth_ == 1;
  auto directly_store = is_src_expr_root && dst_scope_ == "local";
  auto type = Type2String(t.element_of());
  int lanes = t.lanes();
  int bytes = t.lanes() * t.bytes();

  std::string dst, src0, src1, opt;
  const auto *lv = lhs.as<BroadcastNode>(), *rv = rhs.as<BroadcastNode>();
  if (bang_stream_binary_const_ops().count(op) && (lv || rv)) {
    opt = bang_stream_binary_const_ops().at(op);
    src0 = PrintExpr(lv ? rhs : lhs);
    src1 = PrintExpr(lv ? lv->value : rv->value);
  } else {
    opt = bang_stream_binary_ops().at(op);
    src0 = PrintExpr(lhs), src1 = PrintExpr(rhs);
  }

  if (directly_store) {
    already_stored_ = true;
    dst = dst_buff_;
  } else {
    dst = GetUniqueName("tmp");
    GenNRAMTmpBuf(type, dst, bytes);
    os << "(";
  }
  fmt::print(os, "{}_<{}, {}>({}, {}, {})",
             opt, type, lanes, dst, src0, src1);
  if (!directly_store)
    os << ", " << dst << ")";
}
void CodeGenBANG::PrintType(DataType t, std::ostream &os) {
  if (!type_scope_.empty()) {
    os << type_scope_.back();
    return;
  }
  if (t.is_int() || t.is_uint()) {
    switch (t.bits()) {
    case 8:
    case 16:
    case 32: os << (t.is_int() ? "int" : "uint") << t.bits() << "_t";
      return;
    }
  } else if (t.is_float()) {
    switch (t.bits()) {
    case 16: os << "half";
      return;
    case 32: os << "float";
      return;
    }
  } else if (t.is_bool()) {
    os << "bool";
    return;
  } else if (t.is_handle()) {
    os << "void *";
    return;
  }
  LOG(FATAL) << "Cannot convert type " << t << " to BANG type!";
}
void CodeGenBANG::BindThreadIndex(const IterVar &iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = par_var_map().at(iv->thread_tag);
  par_var_dim_.at(iv->thread_tag) = temp_thread_extent_;
  used_par_var_.insert(iv->thread_tag);
}
void CodeGenBANG::VisitExpr_(const BroadcastNode *op, std::ostream &os) {
  auto is_src_expr_root = is_src_expr_ && src_expr_depth_ == 1;
  auto type = Type2String(op->dtype.element_of());
  int lanes = op->lanes;
  int bytes = lanes * op->dtype.bytes();
  auto value = PrintExpr(op->value);
  std::string dst, builtin;
  if (is_src_expr_root) {
    already_stored_ = true;
    builtin = fmt::format("__{}set", scope_map().at(dst_scope_));
    dst = dst_buff_;
  } else {
    builtin = "__nramset";
    dst = GetUniqueName("tmp");
    GenNRAMTmpBuf(type, dst, bytes);
  }
  if (!is_src_expr_root)
    os << "(";
  fmt::print(os, "{}_<{}, {}>({}, {})", builtin, type, lanes, dst, value);
  if (!is_src_expr_root)
    os << dst << ")";
}
void CodeGenBANG::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tir::attr::thread_extent) {
    std::ostringstream os;
    os << op->value;
    try {
      temp_thread_extent_ = std::stoi(os.str());
    } catch (std::invalid_argument &) {
      LOG(FATAL) << "Thread-extent is not constant: " << op->value;
    }
  } else if (op->attr_key == tir::attr::thread_loop) {
    if (op->value.defined())
      no_sync_point_ = true;
    GenStmt() << "THREAD_LOOP {\n";
    auto scope = BeginScope();
    PrintStmt(op->body);
    EndScope(scope);
    GenStmt() << "}\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}
void CodeGenBANG::VisitExpr_(const LoadNode *op, std::ostream &os) {
  int lanes = op->dtype.lanes();
  if (lanes == 1) {
    CodeGenC::VisitExpr_(op, os);
  } else {
    // vectorized load
    CHECK(is_one(op->predicate))
      << "predicated load is not supported";
    const auto *ramp = op->index.as<RampNode>();
    const auto *stride = ramp ? ramp->stride.as<IntImmNode>() : nullptr;
    if (stride && stride->value == 1) {
      auto is_src_expr_root = is_src_expr_ && src_expr_depth_ == 1;
      auto scope = alloc_storage_scope_.count(op->buffer_var.get()) ?
                   alloc_storage_scope_.at(op->buffer_var.get()) : "global";
      auto ref = GetBufferRef(op->dtype, op->buffer_var.get(), ramp->base);
      auto type = Type2String(op->dtype.element_of());
      if (is_src_expr_root) {
        fmt::print(os, "__memcpy({}, {}, sizeof({}) * {}, {})",
                   dst_buff_, ref, type, ramp->lanes, MoveDir(scope, dst_scope_));
        already_stored_ = true;
      } else {
        if (scope != "global") {
          os << "(" << ref << ")";
        } else {
          auto tmp_vid = GetUniqueName("tmp");
          int bytes = op->dtype.bytes() * lanes;
          GenNRAMTmpBuf(type, tmp_vid, bytes);
          fmt::print(os, "(__memcpy({}, {}, sizeof({}) * {}, {}), {})",
                     tmp_vid, ref, type, ramp->lanes, MoveDir(scope, "local"), tmp_vid);
        }
      }
    } else {
      LOG(FATAL) << "not yet implemented";
    }
  }
}
void CodeGenBANG::VisitStmt_(const StoreNode *op) {
  int lanes = op->value->dtype.lanes();
  auto dtype = op->value->dtype;
  if (lanes == 1) {
    CodeGenC::VisitStmt_(op);
  } else {
    // vectorized store
    CHECK(is_one(op->predicate))
      << "predicated store is not supported";
    const auto *ramp = op->index.as<RampNode>();
    const auto *stride = ramp ? ramp->stride.as<IntImmNode>() : nullptr;
    if (stride && stride->value == 1) {
      nram_tmp_buf_.enter_scope();
      auto dst = GetBufferRef(dtype, op->buffer_var.get(), ramp->base);
      auto scope = alloc_storage_scope_.count(op->buffer_var.get()) ?
                   alloc_storage_scope_.at(op->buffer_var.get()) : "global";

      dst_buff_ = dst, dst_scope_ = scope;
      is_src_expr_ = true, src_expr_depth_ = 0;
      already_stored_ = false;
      auto src = PrintExpr(op->value);
      is_src_expr_ = false;

      if (already_stored_) {
        GenStmt() << src << ";\n";
      } else {
        auto type = Type2String(op->value.dtype().element_of());
        fmt::print(GenStmt(), "__memcpy({}, {}, sizeof({}) * {}, {});\n",
                   dst_buff_, src, type, ramp->lanes, MoveDir("local", scope));
      }
      nram_tmp_buf_.leave_scope();
    } else {
      LOG(FATAL) << "not yet implemented";
    }
  }
}
std::string CodeGenBANG::GetBufferRef(DataType t, const VarNode *buffer, PrimExpr index) {
  int lanes = t.lanes();
  if (lanes == 1) {
    return CodeGenC::GetBufferRef(t, buffer, index);
  } else {
    auto buf_vid = GetVarID(buffer);
    auto type = Type2String(t.element_of());
    return fmt::format("(({} *){} + {})", type, buf_vid, PrintExpr(index));
  }
}
void CodeGenBANG::VisitStmt_(const AllocateNode *op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  auto constant_size = op->constant_allocation_size() * op->dtype.lanes();
  CHECK_GT(constant_size, 0)
    << "Can only handle constant size stack allocation for now";
  auto type = Type2String(op->dtype);
  auto scope = alloc_storage_scope_.at(op->buffer_var.get());
  if (no_sync_point_) {
    fmt::print(GenStmt(), "__{}__ {} {}[{}];\n", ScopeAbbrev(scope), type, vid, constant_size);
  } else {
    if (scope == "shared") {
      fmt::print(GenStmt(), "__nram__ {} {}[{}];\n", type, vid, constant_size);
    } else {
      auto mem_vid = GetUniqueName(vid + "_mem");
      fmt::print(GenStmt(),
                 "__{}__ {} {}[N_THREAD * {}];\n",
                 ScopeAbbrev(scope),
                 type,
                 mem_vid,
                 constant_size);
      fmt::print(GenStmt(), "#define {} ({} + THREAD_IDX * {})\n", vid, mem_vid, constant_size);
    }
  }
  RegisterHandleType(op->buffer_var.get(), op->dtype);
  PrintStmt(op->body);
}
int CodeGenBANG::GetTaskDim() {
  GetTaskDim3();
  int res = 1;
  for (auto d : task_dim_)
    res *= d;
  return res;
}
std::array<int, 3> CodeGenBANG::GetTaskDim3() {
  if (task_dim_[0] == 0) {
    task_dim_.fill(1);
    for (const auto &pvd: par_var_dim_) {
      if (pvd.first[0] == 'b')
        task_dim_[0] *= pvd.second;
    }
  }
  return task_dim_;
}
void CodeGenBANG::PrintFuncPrefix() {
  stream << "extern \"C\" __mlu_entry__ void";
}
void CodeGenBANG::PreFunctionBody(const PrimFunc &f) {
  GenStmt() << "PRE_DECL\n";
}
void CodeGenBANG::InitFuncState(const PrimFunc &f) {
  CHECK(!already_gen_kernel_)
    << "Cannot generate more than one kernel";
  already_gen_kernel_ = true;
  nram_tmp_buf_.init(GetUniqueName("nram_tmp_buf"));
  CodeGenC::InitFuncState(f);
}
void CodeGenBANG::AddFunction(const PrimFunc &f) {
  auto body = tir::ModifyParallelModel(f->body);
  PrimFunc ff(f->params, body, f->ret_type, f->buffer_map, f->attrs);
  CodeGenC::AddFunction(ff);
}
void CodeGenBANG::VisitStmt_(const ForNode *op) {
  CHECK(is_const_int(op->min, 0));
  if (op->for_type == tir::ForType::Unrolled) {
    GenStmt() << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}
void CodeGenBANG::VisitExpr(const PrimExpr &n, std::ostream &os) {
  src_expr_depth_++;
  CodeGenC::VisitExpr(n, os);
}
void CodeGenBANG::VisitExpr_(const CallNode *op, std::ostream &os) {
  if ((op->call_type == CallNode::Extern
      || op->call_type == CallNode::PureExtern)
      && op->name == "__bang_update_with") {
    auto bytes = op->dtype.bytes();
    auto lanes = op->dtype.lanes();
    auto type = Type2String(op->dtype);

    type_scope_.push_back(type);
    auto dst_buf = PrintExpr(op->args[0]);
    type_scope_.pop_back();

    auto acc_fun = op->args[1].as<StringImmNode>()->value;
    auto src_fun = op->args[2].as<StringImmNode>()->value;
    auto tmp_buf = GetUniqueName("tmp");
    GenNRAMTmpBuf(type, tmp_buf, lanes * bytes);

    os << "(" << src_fun << "(" << tmp_buf;
    int n_args = op->args.size();
    for (int i = 3; i < n_args; ++i)
      os << ", " << PrintExpr(op->args[i]);
    os << "), ";
    fmt::print(os, "__bang_{}_<{}, {}>({}, {}, {}))",
               acc_fun, type, lanes, dst_buf, dst_buf, tmp_buf);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

}
}