//
// Created by herlight on 2020/3/7.
//

#include "codegen_bang.h"
#include "../../arith/compute_expr.h"
#include "./modify_parallel_model.h"
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/expr_functor.h>
#include "../../arith/pattern_match.h"

namespace tvm {
namespace codegen {

CodeGenBANG::CodeGenBANG() {
  restrict_keyword_ = "__restrict__";
}
std::string CodeGenBANG::Finish() {
  decl_stream
      << "#include \"mlu.h\"\n"
      << "#include <cstring>\n";
  decl_stream
      << "#define DECL_MEMSET(name) "
      << "template <typename T, int VecN, int TotN> "
      << "__mlu_func__ void name ## _ (T *ptr, T val) "
      << "{ name (ptr, VecN, val); for (int i = VecN; i < TotN; ++i) ptr[i] = val; }\n";
  for (const auto &on : bang_memset()) {
    decl_stream
        << "DECL_MEMSET(" << on << ")\n";
  }
  decl_stream
      << "#define DECL_STRM_OP(name, op) "
      << "template <typename T, int VecN, int TotN> "
      << "__mlu_func__ void name ## _ (T *dst, T *src0, T *src1) "
      << "{ name (dst, src0, src1, VecN) ; for (int i = VecN; i < TotN; ++i) dst[i] = src0[i] op src1[i]; }\n";
  for (const auto &on : bang_stream_op()) {
    decl_stream
        << "DECL_STRM_OP(" << on.second << ", " << on.first << ")\n";
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
      << "#define PAR_VAR_DECL "
      << "if (taskIdX >= N_BLOCK) return; "
      << (used_par_var_.count("blockIdx.x") ? "int blockIdx_x = BLOCK_IDX_X; " : "")
      << (used_par_var_.count("blockIdx.y") ? "int blockIdx_y = BLOCK_IDX_Y; " : "")
      << (used_par_var_.count("blockIdx.z") ? "int blockIdx_z = BLOCK_IDX_Z; " : "")
      << '\n';
  decl_stream
      << "#define N_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)\n"
      << "#define N_THREAD (THREAD_DIM_X * THREAD_DIM_Y * THREAD_DIM_Z)\n";

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

  decl_stream
      << "#define THREAD_IDX ("
      << (used_par_var_.count("threadIdx.x") ? "threadIdx_x" : "0") << " * (THREAD_DIM_Y * THREAD_DIM_Z) + "
      << (used_par_var_.count("threadIdx.y") ? "threadIdx_y" : "0") << " * THREAD_DIM_Z + "
      << (used_par_var_.count("threadIdx.z") ? "threadIdx_z" : "0") << ")\n";

  decl_stream
      << "#define VEC_STORE \n";

  return CodeGenC::Finish();
}
void CodeGenBANG::PrintStorageSync(const CallNode *op) {
}
void CodeGenBANG::PrintStorageScope(const std::string &scope, std::ostream &os) {
}
void CodeGenBANG::PrintVecBinaryOp(const std::string &op, DataType t,
                                   PrimExpr lhs, PrimExpr rhs, std::ostream &os) {
  auto is_src_expr_root = is_src_expr_ && src_expr_depth_ == 1;
  int lanes = t.lanes();
  auto type = Type2String(t.element_of());
  auto opt = bang_stream_op().at(op);
  auto src0 = PrintExpr(lhs), src1 = PrintExpr(rhs);
  if (is_src_expr_root && dst_scope_ != "global") {
    already_stored_ = true;
    os << opt << "_<" << type << ", " << (lanes / 64 * 64) << ", " << lanes << ">("
       << dst_buff_ << ", " << src0 << ", " << src1 << ")";
  } else {
    auto dst = GetUniqueName("tmp");
    GenStmt() << type << " " << dst << '[' << lanes << "];\n";
    os << "("
       << opt << "_<" << type << ", " << (lanes / 64 * 64) << ", " << lanes << ">("
       << dst << ", " << src0 << ", " << src1 << "), "
       << dst << ")";
  }
}
void CodeGenBANG::PrintType(DataType t, std::ostream &os) {
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
  auto value = PrintExpr(op->value);
  if (is_src_expr_root) {
    already_stored_ = true;
    os << "__gdramset_<" << type << ", " << (lanes / 64 * 64) << ", " << lanes << ">("
       << dst_buff_ << ", " << value << ")";
  } else {
    auto dst = GetUniqueName("tmp");
    GenStmt() << type << " " << dst << '[' << lanes << "];\n";
    os << "("
       << "__nramset_<" << type << ", " << (lanes / 64 * 64) << ", " << lanes << ">("
       << dst << ", " << value << "), "
       << dst << ")";
  }
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
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, op->dtype.lanes()).Match(op->index)) {
      auto is_src_expr_root = is_src_expr_ && src_expr_depth_ == 1;
      auto scope = alloc_storage_scope_.count(op->buffer_var.get()) ?
                   alloc_storage_scope_.at(op->buffer_var.get()) : "global";
      auto ref = GetBufferRef(op->dtype, op->buffer_var.get(), base.Eval());
      auto type = Type2String(op->dtype.element_of());
      if (is_src_expr_root) {
        os << "__memcpy(" << dst_buff_ << ", " << ref << ", sizeof(" << type << ") * " << lanes
           << ", " << MoveDir(scope, dst_scope_) << ")";
        already_stored_ = true;
      } else {
        if (scope != "global") {
          os << "(" << ref << ")";
        } else {
          auto tmp_vid = GetUniqueName("tmp");
          GenStmt() << type << " " << tmp_vid << '[' << lanes << "];\n";
          os << "(__memcpy(" << tmp_vid << ", " << ref << ", sizeof(" << type << ") * " << lanes
             << ", " << MoveDir(scope, "local") << "), "
             << tmp_vid << ")";
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
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, dtype.lanes()).Match(op->index)) {
      GenStmt() << "VEC_STORE {\n";
      auto mark = BeginScope();
      auto dst = GetBufferRef(dtype, op->buffer_var.get(), base.Eval());
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
        GenStmt() << "__memcpy(" << dst_buff_ << ", " << src << ", sizeof(" << type << ") * " << lanes
                  << ", " << MoveDir("local", scope) << ");\n";
      }
      EndScope(mark);
      GenStmt() << "}\n";
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
    std::ostringstream os;
    auto buf_vid = GetVarID(buffer);
    auto type = Type2String(t.element_of());
    os << "((" << type << " *)" << buf_vid << " + (" << PrintExpr(index) << "))";
    return os.str();
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
    CHECK_NE(scope, "global");
    GenStmt() << type << " " << vid << '[' << constant_size << "];\n";
  } else {
    if (scope == "shared") {
      GenStmt() << type << " " << vid << '[' << constant_size << "];\n";
    } else if (scope == "local") {
      auto mem_vid = GetUniqueName(vid + "_mem");
      GenStmt() << type << " " << mem_vid << '[' << "N_THREAD * " << constant_size << "];\n";
      GenStmt() << "#define " << vid << " (" << mem_vid << " + THREAD_IDX * " << constant_size << ")\n";
    } else {
      LOG(FATAL) << "BANG kernel cannot support memory scope: " << scope;
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
  GenStmt() << "PAR_VAR_DECL\n";
}
void CodeGenBANG::InitFuncState(const PrimFunc &f) {
  CHECK(!already_gen_kernel_)
    << "Cannot generate more than one kernel";
  already_gen_kernel_ = true;
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

}
}