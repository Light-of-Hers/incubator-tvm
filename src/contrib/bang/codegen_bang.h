//
// Created by herlight on 2020/3/7.
//

#ifndef TVM_SRC_CONTRIB_BANG_CODEGEN_BANG_H_
#define TVM_SRC_CONTRIB_BANG_CODEGEN_BANG_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <string>
#include <unordered_map>
#include <map>
#include <array>
#include <set>
#include "../../target/source/codegen_c.h"
#include "./fmt/ostream.h"

namespace tvm {
namespace codegen {

class CodeGenBANG final : public CodeGenC {
public:
  CodeGenBANG();

  // overwrite method
  std::string Finish();
  void AddFunction(const PrimFunc &f);

  // override behavior
  void PrintFuncPrefix() override;
  void PreFunctionBody(const PrimFunc &f) override;
  void InitFuncState(const PrimFunc &f) override;
  void PrintStorageSync(const CallNode *op) override;
  void PrintStorageScope(const std::string &scope, std::ostream &os) override;
  void PrintVecBinaryOp(
      const std::string &op, DataType t,
      PrimExpr lhs, PrimExpr rhs, std::ostream &os) override;
  void PrintType(DataType t, std::ostream &os) override;
  void BindThreadIndex(const IterVar &iv) override;

  // overload visitor
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) override;
  void VisitStmt_(const AttrStmtNode *op) override;
  void VisitStmt_(const AllocateNode *op) override;
  void VisitStmt_(const ForNode *op) override;
  void VisitStmt_(const StoreNode *op) override;
  void VisitExpr_(const LoadNode *op, std::ostream &os) override;

  void VisitExpr(const PrimExpr &n, std::ostream &os) override;

  // other methods
  int GetTaskDim();
  std::array<int, 3> GetTaskDim3();

protected:
  std::string GetBufferRef(DataType t, const VarNode *buffer, PrimExpr index) override;

private:

  int temp_thread_extent_{};
  std::map<std::string, int> par_var_dim_{
      {"blockIdx.x", 1},
      {"blockIdx.y", 1},
      {"blockIdx.z", 1},
      {"threadIdx.x", 1},
      {"threadIdx.y", 1},
      {"threadIdx.z", 1}
  };
  std::set<std::string> used_par_var_;
  static inline const std::map<std::string, std::string> &par_var_map() {
    static const std::map<std::string, std::string> ret{
        {"blockIdx.x", "blockIdx_x"},
        {"blockIdx.y", "blockIdx_y"},
        {"blockIdx.z", "blockIdx_z"},
        {"threadIdx.x", "threadIdx_x"},
        {"threadIdx.y", "threadIdx_y"},
        {"threadIdx.z", "threadIdx_z"}
    };
    return ret;
  }
  static inline const std::map<std::string, std::string> &bang_stream_binary_ops() {
    static const std::map<std::string, std::string> ret{
        {"+", "__bang_add"},
        {"-", "__bang_sub"},
        {"*", "__bang_mul"}
    };
    return ret;
  }
  static inline const std::map<std::string, std::string> &bang_stream_binary_const_ops() {
    static const std::map<std::string, std::string> ret{
        {"*", "__bang_mul_const"},
        {"+", "__bang_add_const"},
        {"-", "__bang_sub_const"},
    };
    return ret;
  }
  static inline const std::map<std::string, std::string> &scope_map() {
    static const std::map<std::string, std::string> res = {
        {"global", "gdram"},
        {"local", "nram"},
        {"local.wram", "wram"},
    };
    return res;
  }

  bool is_src_expr_{false};
  int src_expr_depth_{0};
  bool already_stored_{false};
  std::string dst_buff_;

  std::string dst_scope_;
  bool already_gen_kernel_{false};
  bool no_sync_point_{false};

  std::array<int, 3> task_dim_{0, 0, 0};

  struct {
    inline std::string name() const { return name_; }
    inline int max_size() const { return max_size_; }
    void init(const std::string &nam) {
      name_ = nam;
      max_size_ = 0;
      scope_ = {0};
    }
    inline void enter_scope() { scope_.push_back(scope_.back()); }
    inline void leave_scope() { scope_.pop_back(); }
    inline int alloc(int bytes) {
      int aligned = (scope_.back() + 63) / 64 * 64;
      scope_.back() = aligned + bytes;
      max_size_ = std::max(max_size_, aligned + bytes);
      return aligned;
    }
  private:
    std::string name_{};
    int max_size_{};
    std::vector<int> scope_{};
  } nram_tmp_buf_{};
  inline void GenNRAMTmpBuf(const std::string &type,
                            const std::string &buf, int bytes) {
    fmt::print(GenStmt(),
               "{} *{} = ({} *)({} + {});\n",
               type, buf, type,
               nram_tmp_buf_.name(), nram_tmp_buf_.alloc(bytes));
  }
  inline std::ostream &GenStmt() {
    PrintIndent();
    return stream;
  }
  inline std::string Type2String(DataType t) {
    std::ostringstream os;
    PrintType(t, os);
    return os.str();
  }
  inline std::string MoveDir(const std::string &from_scope, const std::string &to_scope) {
    auto to_upper = [](std::string s) -> std::string {
      std::for_each(s.begin(), s.end(), [](char &c) { c = std::toupper(c); });
      return s;
    };
    return to_upper(scope_map().at(from_scope)) + "2" + to_upper(scope_map().at(to_scope));
  }
  inline std::string ScopeAbbrev(const std::string &scope) {
    if (scope_map().count(scope) == 0)
      LOG(FATAL) << "BANG kernel cannot support memory scope: " << scope;
    return scope_map().at(scope);
  }
};

}
}

#endif //TVM_SRC_CONTRIB_BANG_CODEGEN_BANG_H_
