/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/transforms/collective_select_folder.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using SourceTargetPair = std::pair<int64_t, int64_t>;
using SourceTargetPairs = std::vector<SourceTargetPair>;

struct SelectPredInfo {
  const bool is_replica_select;
  int64_t constant;
  Comparison::Direction direction;
  HloOpcode device_id_type;  // replica_id or partition_id
  HloInstruction* true_operand;
  HloInstruction* false_operand;

  explicit SelectPredInfo(bool is_replica_or_select)
      : is_replica_select(is_replica_or_select) {}

  SelectPredInfo(bool is_replica_or_select, int64_t constant,
                 Comparison::Direction direction, HloOpcode device_id_type,
                 HloInstruction* true_operand, HloInstruction* false_operand)
      : is_replica_select(is_replica_or_select),
        constant(constant),
        direction(direction),
        device_id_type(device_id_type),
        true_operand(true_operand),
        false_operand(false_operand) {}
};

// Returns handy references to %constant, %true_operand, %false_operand of the
// select(broadcast(compare(current_device_id, constant)))
// or
// select(compare(current_device_id, constant))
SelectPredInfo GetPredSelectInfo(HloInstruction* select) {
  if (select->opcode() != HloOpcode::kSelect) {
    return SelectPredInfo(false);
  }

  const HloCompareInstruction* compare;
  if (select->operand(0)->opcode() == HloOpcode::kCompare) {
    compare = Cast<HloCompareInstruction>(select->operand(0));
  } else if (select->operand(0)->opcode() == HloOpcode::kBroadcast &&
             select->operand(0)->operand(0)->opcode() == HloOpcode::kCompare) {
    compare = Cast<HloCompareInstruction>(select->operand(0)->operand(0));
  } else {
    return SelectPredInfo(false);
  }

  bool is_replica_or_partition_compare =
      (compare->operand(0)->opcode() == HloOpcode::kReplicaId ||
       compare->operand(0)->opcode() == HloOpcode::kPartitionId) &&
      compare->operand(1)->opcode() == HloOpcode::kConstant;

  if (!is_replica_or_partition_compare) return SelectPredInfo(false);

  int64_t id_value =
      compare->operand(1)->literal().GetFirstInteger().value_or(-1);

  return SelectPredInfo(true, id_value, compare->direction(),
                        compare->operand(0)->opcode(),
                        select->mutable_operand(1), select->mutable_operand(2));
}

bool IsOnlySender(int64_t device_id, const SourceTargetPairs& pairs) {
  if (pairs.size() == 1 && pairs[0].first == device_id) return true;
  return false;
}

bool IsNotPresent(int64_t device_id, const SourceTargetPairs& pairs) {
  for (const auto& pair : pairs) {
    if (pair.first == device_id) return false;
  }
  return true;
}

absl::StatusOr<bool> update(HloInstruction* cp, HloInstruction* data) {
  TF_RETURN_IF_ERROR(cp->ReplaceOperandWith(0, data));
  return true;
}

// Recognizer the pattern and update if applicable.
absl::StatusOr<bool> TryFoldSelect(HloInstruction* in) {
  if (in->opcode() != HloOpcode::kCollectivePermute) return false;
  auto select_info = GetPredSelectInfo(in->mutable_operand(0));
  if (!select_info.is_replica_select) return false;

  HloCollectivePermuteInstruction* cp =
      Cast<HloCollectivePermuteInstruction>(in);

  // We have to maintain integrity of relationship between partition/replica
  // and collective-permute's channel_id.
  // That is we can only fold select when
  // 1. cp has channel_id and condition is based on partition_id
  // 2. cp has no channel_id and condition is based on replica_id
  // see enum class CollectiveOpGroupMode for details.
  if ((select_info.device_id_type == HloOpcode::kPartitionId &&
       !cp->channel_id().has_value()) ||
      (select_info.device_id_type == HloOpcode::kReplicaId &&
       cp->channel_id().has_value())) {
    return false;
  }

  auto device_id = select_info.constant;
  auto pairs = cp->source_target_pairs();
  if (select_info.direction == Comparison::Direction::kEq) {
    if (IsOnlySender(device_id, pairs)) {
      return update(cp, select_info.true_operand);
    } else if (IsNotPresent(device_id, pairs)) {
      return update(cp, select_info.false_operand);
    }
  }

  if (select_info.direction == Comparison::Direction::kNe) {
    if (IsNotPresent(device_id, pairs)) {
      return update(cp, select_info.true_operand);
    } else if (IsOnlySender(device_id, pairs)) {
      return update(cp, select_info.false_operand);
    }
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> CollectiveSelectFolder::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, TryFoldSelect(instruction));
      changed |= local_changed;
    }
  }
  return changed;
}

}  // namespace xla
