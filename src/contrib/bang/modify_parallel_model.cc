//
// Created by herlight on 2020/3/28.
//

#include "modify_parallel_model.h"
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/expr.h>
#include "../../tir/ir/functor_common.h"
#include <vector>

namespace tvm {
namespace tir {

class SyncPointsDetector final : public StmtVisitor {
public:
  std::vector<Stmt> DetectSyncPoints(const Stmt &n) {
    sync_points_.clear();
    VisitStmt(n);
    return sync_points_;
  }
protected:
  void VisitStmt_(const EvaluateNode *op) override {
    const auto call = op->value.as<CallNode>();
    if (call && call->is_intrinsic(intrinsic::tvm_storage_sync)) {
      sync_points_.push_back(GetRef<Stmt>(op));
    }
    StmtVisitor::VisitStmt_(op);
  }
private:
  std::vector<Stmt> sync_points_{};
};

class ThreadLoopMutator final : public StmtMutator {
public:
  Stmt MutateThreadLoop(const Stmt &n, const Stmt &sp) {
    keep_splitting_ = false;
    sync_point_ = sp;
    return VisitStmt(n);
  }
protected:
  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == attr::thread_loop) {
      keep_splitting_ = false;
      auto body = VisitStmt(op->body);
      if (keep_splitting_) {
        keep_splitting_ = false;
        return body;
      } else {
        auto n = CopyOnWrite(op);
        n->body = body;
        return Stmt(n);
      }
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const IfThenElseNode *op) override {
    keep_splitting_ = false;
    auto then_case = VisitStmt(op->then_case);
    auto split_then = keep_splitting_;
    keep_splitting_ = false;
    auto else_case = op->else_case.defined() ? VisitStmt(op->else_case) : Stmt{};
    auto split_else = keep_splitting_;
    auto n = CopyOnWrite(op);
    n->condition = op->condition;
    if (split_then) {
      n->then_case = std::move(then_case);
      n->else_case = AttrStmtNode::make({}, attr::thread_loop, {}, else_case);
      keep_splitting_ = true;
    } else if (split_else) {
      n->else_case = std::move(else_case);
      n->then_case = AttrStmtNode::make({}, attr::thread_loop, {}, then_case);
      keep_splitting_ = true;
    } else {
      n->then_case = std::move(then_case);
      n->else_case = std::move(else_case);
      keep_splitting_ = false;
    }
    return Stmt(n);
  }
  Stmt VisitStmt_(const SeqStmtNode *op) override {
    Array<Stmt> seq;
    int split_pos = -1;
    for (size_t i = 0; i < op->size(); ++i) {
      keep_splitting_ = false;
      seq.push_back(VisitStmt((*op)[i]));
      if (keep_splitting_)
        split_pos = i;
    }
    if (split_pos >= 0) {
      Array<Stmt> before, after, all;
      for (int i = 0; i < split_pos; ++i)
        before.push_back(seq[i]);
      for (size_t i = split_pos + 1; i < seq.size(); ++i)
        after.push_back(seq[i]);
      if (!before.empty())
        all.push_back(AttrStmtNode::make({}, attr::thread_loop, {}, SeqStmt(before)));
      all.push_back(seq[split_pos]);
      if (!after.empty())
        all.push_back(AttrStmtNode::make({}, attr::thread_loop, {}, SeqStmt(after)));
      keep_splitting_ = true;
      return SeqStmt(all);
    } else {
      auto n = CopyOnWrite(op);
      n->seq = seq;
      keep_splitting_ = false;
      return Stmt(n);
    }
  }
  Stmt VisitStmt_(const EvaluateNode *op) override {
    if (sync_point_.same_as(GetRef<Stmt>(op)))
      keep_splitting_ = true;
    return StmtMutator::VisitStmt_(op);
  }
private:
  Stmt sync_point_;
  bool keep_splitting_{false};
};

Stmt ModifyParallelModel(const Stmt &stmt) {
  SyncPointsDetector spd;
  ThreadLoopMutator tlm;
  auto sync_points = spd.DetectSyncPoints(stmt);
  if (sync_points.empty()) {
    return AttrStmtNode::make({}, attr::thread_loop,
                              StringImmNode::make("no_sync_point"), stmt);
  } else {
    auto res = stmt;
    for (const auto &sp: sync_points) {
      res = tlm.MutateThreadLoop(res, sp);
    }
    return res;
  }
}

}
}