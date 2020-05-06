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
  static inline const std::map<std::string, std::string> &bang_stream_op() {
    static const std::map<std::string, std::string> ret{
        {"+", "__bang_add"},
        {"-", "__bang_sub"},
        {"*", "__bang_mul"}
    };
    return ret;
  }
  static inline const std::set<std::string> &bang_memset() {
    static const std::set<std::string> ret{
        "__nramset", "__gdramset", "__ldramset"
    };
    return ret;
  }

  bool is_src_expr_{false};
  int src_expr_depth_{0};
  bool already_stored_{false};
  std::string dst_buff_;
  std::string dst_scope_;

  bool already_gen_kernel_{false};
  bool no_sync_point_{false};
  std::array<int, 3> task_dim_{0, 0, 0};

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
    auto get_scope = [](const std::string &scope) -> std::string {
      if (scope == "global") {
        return "GDRAM";
      } else {
        return "NRAM";
      }
    };
    return get_scope(from_scope) + "2" + get_scope(to_scope);
  }
};

}
}

#endif //TVM_SRC_CONTRIB_BANG_CODEGEN_BANG_H_
