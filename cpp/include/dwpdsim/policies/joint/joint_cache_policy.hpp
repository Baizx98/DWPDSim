#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "dwpdsim/policies/memory/memory_policy.hpp"
#include "dwpdsim/policies/storage/storage_policy.hpp"
#include "dwpdsim/radix_tree.hpp"
#include "dwpdsim/types.hpp"

namespace dwpdsim {

enum class JointActionKind : std::uint8_t {
    CleanEvict,
    Drop,
    MoveToSlc,
    MoveToTlc,
};

struct JointEvictionPlan {
    JointActionKind kind = JointActionKind::Drop;
    // Ordered top-to-bottom (parent before child) for writes and memory removal.
    std::vector<NodeId> memory_nodes;
    StorageTier target_tier = StorageTier::Slc;
    std::uint32_t stream_id = 0;
    // Ordered bottom-to-top (leaf before parent) for prefix-valid SSD TRIM before write.
    std::vector<NodeId> ssd_trim_nodes;
};

class JointCachePolicy {
  public:
    virtual ~JointCachePolicy() = default;

    virtual void bind_tree(const RadixTree& tree) = 0;
    virtual bool admit_storage_hit(
        const AccessContext& access,
        const Node& node,
        const RadixTree& tree
    ) const = 0;
    virtual void on_request_begin(
        const RequestContext& request,
        const StorageView& storage
    ) = 0;
    virtual void on_request_end(
        const RequestContext& request,
        const StorageView& storage,
        std::uint64_t hit_blocks,
        std::uint64_t miss_blocks
    ) = 0;
    virtual void on_node_created(NodeId node_id, const RadixTree& tree) = 0;
    virtual void on_node_pruned(
        NodeId node_id,
        std::optional<NodeId> parent_id,
        const RadixTree& tree
    ) = 0;
    virtual void on_memory_commit(
        const MemoryMutation& mutation,
        const RadixTree& tree
    ) = 0;
    virtual void on_storage_commit(
        const StorageMutation& mutation,
        const StorageView& storage_after_commit
    ) = 0;
    virtual JointEvictionPlan plan_eviction(
        const AccessContext& context,
        const StorageView& storage
    ) const = 0;
    virtual std::vector<std::pair<StorageTier, NodeId>> collect_proactive_trims(
        const RequestContext& request,
        const StorageView& storage
    ) const {
        static_cast<void>(request);
        static_cast<void>(storage);
        return {};
    }
    virtual StoragePolicyStats stats(const StorageView& storage) const = 0;
};

}  // namespace dwpdsim
