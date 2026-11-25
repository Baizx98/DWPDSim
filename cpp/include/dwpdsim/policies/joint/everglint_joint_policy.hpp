#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dwpdsim/policies/joint/joint_cache_policy.hpp"

namespace dwpdsim {

struct EverglintJointPolicyConfig {
    // "everglint" | "gandalf_tlc" | "gandalf_slc" | "gandalf_wear_share"
    std::string mode = "everglint";
    bool admit_storage_hits = false;
    double wear_lambda = 1.0e8;
    double wear_eta = 0.1;
    std::uint64_t slc_physical_bytes = 0;
    std::uint64_t tlc_physical_bytes = 0;
    double slc_pe_limit = 30000.0;
    double tlc_pe_limit = 3000.0;
    double slc_waf_est = 1.0;
    double tlc_waf_est = 1.0;
    std::uint64_t horizon_requests = 4096;
    std::uint64_t chunk_blocks = 16;
    std::uint64_t candidate_limit = 32;
    std::uint64_t victim_limit = 64;
    bool use_survival = true;
    bool use_prefix_cost = true;
    bool allow_drop = true;
    bool use_value_victim = true;
    double prior_reuse_prob = 0.5;
    double prior_weight = 10.0;
    double residence_ema_alpha = 0.05;
    double logical_fill_fraction = 1.0;
    double trim_gamma = 0.25;
    // Fine-grained ablation, stability, promotion, and closed-loop control knobs
    bool use_cadence_strata = true;
    bool use_jensen_exact = true;
    bool use_session_coherence = true;
    bool use_smooth_sigmoid = true;
    double session_hysteresis = 0.05;
    std::uint64_t promote_hit_threshold = 0;
    std::uint64_t gandalf_hash_salt = 47;
    double target_life_years = 0.0;
    double dual_step_kappa = 0.0;
    std::uint64_t dual_window_ns = 900ULL * 1000000000ULL;
};

class EverglintJointPolicy final : public JointCachePolicy {
  public:
    explicit EverglintJointPolicy(EverglintJointPolicyConfig config);

    void bind_tree(const RadixTree& tree) override;
    bool admit_storage_hit(
        const AccessContext& access,
        const Node& node,
        const RadixTree& tree
    ) const override;
    void on_request_begin(
        const RequestContext& request,
        const StorageView& storage
    ) override;
    void on_request_end(
        const RequestContext& request,
        const StorageView& storage,
        std::uint64_t hit_blocks,
        std::uint64_t miss_blocks
    ) override;
    void on_node_created(NodeId node_id, const RadixTree& tree) override;
    void on_node_pruned(
        NodeId node_id,
        std::optional<NodeId> parent_id,
        const RadixTree& tree
    ) override;
    void on_memory_commit(
        const MemoryMutation& mutation,
        const RadixTree& tree
    ) override;
    void on_storage_commit(
        const StorageMutation& mutation,
        const StorageView& storage_after_commit
    ) override;
    JointEvictionPlan plan_eviction(
        const AccessContext& context,
        const StorageView& storage
    ) const override;
    std::vector<std::pair<StorageTier, NodeId>> collect_proactive_trims(
        const RequestContext& request,
        const StorageView& storage
    ) const override;
    StoragePolicyStats stats(const StorageView& storage) const override;

  private:
    static constexpr std::size_t kNumBuckets = 32;
    static constexpr std::size_t kNumStrata = 4;  // 0: first-turn, 1: fast, 2: med, 3: slow
    using RecencyKey = std::pair<std::uint64_t, NodeId>;

    struct ReturnHistogram {
        std::array<double, kNumBuckets> edges{};
        std::array<double, kNumBuckets> returned{};
        std::array<double, kNumBuckets> return_sum{};
        double expired_beyond_h = 0.0;

        void init(double horizon);
        std::size_t bucket_for(double gap) const noexcept;
        void observe_return(double gap, double horizon);
        void observe_expired();
        std::pair<double, double> predict(
            double idle_age,
            double horizon,
            double prior_p,
            double prior_w
        ) const noexcept;
        double predict_survival(
            double idle_age,
            double horizon,
            double residence_scale,
            double prior_p,
            double prior_w,
            bool jensen_exact
        ) const noexcept;
    };

    struct NodeMeta {
        std::uint64_t depth = 1;
        std::uint64_t last_req_seq = 0;
        std::uint64_t write_req_seq = 0;
        std::uint64_t reuse_count = 0;
        AffinityId session_id = 0;
        double expected_context_blocks = 1.0;
        std::uint32_t cached_child_count = 0;
        bool was_cached = false;
        bool in_dram_clean = false;
        bool in_dram_leaf = false;
        bool in_dram_root = false;
        std::array<bool, 2> in_ssd_leaf{false, false};
        std::uint64_t indexed_key_seq = 0;
    };

    struct SessionState {
        std::uint64_t last_req_seq = 0;
        std::uint64_t req_count = 0;
        double ema_gap = 0.0;
        double ema_context_blocks = 1.0;
        bool expired_counted = false;
        std::optional<StorageTier> preferred_tier;
    };

    enum class Mode : std::uint8_t {
        Everglint,
        GandalfTlc,
        GandalfSlc,
        GandalfWearShare,
    };

    static Mode parse_mode(const std::string& mode);
    static std::uint64_t stable_hash(std::uint64_t value, std::uint64_t salt) noexcept;

    void remove_from_all_sets(NodeId id, NodeMeta& meta);
    void refresh_node_sets(NodeId id, const RadixTree& tree);
    void refresh_children_root_membership(NodeId parent_id, const RadixTree& tree);

    double region_budget_bytes(StorageTier tier, const StorageView& storage) const noexcept;
    double effective_waf(StorageTier tier, const StorageView& storage) const noexcept;
    double region_wear(StorageTier tier, const StorageView& storage) const noexcept;
    double tier_minimax_weight(StorageTier tier, const StorageView& storage) const noexcept;
    double wear_potential(double rho_s, double rho_t) const noexcept;
    double marginal_wear(
        StorageTier tier,
        std::uint64_t write_bytes,
        const StorageView& storage
    ) const noexcept;
    double uniform_marginal_wear(
        std::uint64_t write_bytes,
        const StorageView& storage
    ) const noexcept;
    std::size_t session_stratum(AffinityId sid) const noexcept;
    double effective_residence(StorageTier tier, const StorageView& storage) const noexcept;
    double survival_prob(
        const NodeMeta& meta,
        StorageTier tier,
        const StorageView& storage
    ) const noexcept;
    double uniform_survival_prob(
        const NodeMeta& meta,
        const StorageView& storage
    ) const noexcept;
    std::pair<double, double> predict_node(const NodeMeta& meta) const noexcept;
    double block_recompute_cost(const NodeMeta& meta) const noexcept;
    std::uint64_t tier_capacity_limit(StorageTier tier, const StorageView& storage) const noexcept;
    std::uint32_t stream_for(
        StorageTier tier,
        AffinityId affinity_id,
        NodeId endpoint,
        const StorageView& storage
    ) const noexcept;

    struct VictimChain {
        std::vector<NodeId> nodes_bottom_up;
        double loss_per_block = 0.0;
        double total_loss = 0.0;
        std::uint64_t recency = 0;
    };

    std::vector<VictimChain> collect_victim_candidates(
        StorageTier tier,
        const std::unordered_set<NodeId>& protected_set,
        const StorageView& storage
    ) const;

    bool select_ssd_victims(
        StorageTier tier,
        std::uint64_t need_blocks,
        const std::vector<VictimChain>& candidates,
        const std::unordered_set<NodeId>& protected_set,
        const StorageView& storage,
        std::vector<NodeId>& out_victims,
        double& out_replacement_loss
    ) const;

    EverglintJointPolicyConfig config_;
    Mode mode_;
    const RadixTree* tree_ = nullptr;

    std::uint64_t current_req_seq_ = 0;
    AffinityId current_affinity_ = 0;
    double current_req_blocks_ = 1.0;

    // Active lambda (equals config_.wear_lambda unless closed-loop dual control is active)
    double active_wear_lambda_ = 1.0e8;
    std::optional<TimestampNs> dual_window_start_ns_;
    std::array<double, 2> dual_window_start_rho_{0.0, 0.0};

    std::array<ReturnHistogram, kNumStrata> histograms_;
    std::unordered_map<AffinityId, SessionState> sessions_;
    std::set<std::pair<std::uint64_t, AffinityId>> session_recency_;

    std::unordered_map<NodeId, NodeMeta> nodes_;
    std::set<RecencyKey> dram_clean_set_;
    std::set<RecencyKey> dram_leaf_set_;
    std::set<RecencyKey> dram_root_set_;
    std::array<std::set<RecencyKey>, 2> ssd_leaf_set_;

    std::array<std::uint64_t, 2> program_bytes_{0, 0};
    std::array<double, 2> est_nand_bytes_{0.0, 0.0};
    std::array<double, 2> residence_ema_{0.0, 0.0};
    std::array<std::uint64_t, 2> eviction_samples_{0, 0};
    std::array<double, 2> write_blocks_per_req_ema_{0.0, 0.0};
    std::array<std::uint64_t, 2> req_write_blocks_{0, 0};
    double global_residence_ema_ = 0.0;
    std::uint64_t global_eviction_samples_ = 0;
    double global_write_blocks_per_req_ema_ = 0.0;
};

}  // namespace dwpdsim
