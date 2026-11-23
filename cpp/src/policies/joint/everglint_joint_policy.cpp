#include "dwpdsim/policies/joint/everglint_joint_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace dwpdsim {

void EverglintJointPolicy::ReturnHistogram::init(double horizon) {
    const double h = std::max(2.0, horizon);
    for (std::size_t i = 0; i < kNumBuckets; ++i) {
        edges[i] = std::pow(h, static_cast<double>(i + 1) / static_cast<double>(kNumBuckets));
        returned[i] = 0.0;
        return_sum[i] = 0.0;
    }
    edges[kNumBuckets - 1] = h;
    expired_beyond_h = 0.0;
}

std::size_t EverglintJointPolicy::ReturnHistogram::bucket_for(double gap) const noexcept {
    const double g = std::max(1.0, gap);
    const auto it = std::lower_bound(edges.begin(), edges.end(), g);
    if (it == edges.end()) {
        return kNumBuckets - 1;
    }
    return static_cast<std::size_t>(it - edges.begin());
}

void EverglintJointPolicy::ReturnHistogram::observe_return(double gap, double horizon) {
    if (gap <= horizon) {
        const std::size_t b = bucket_for(gap);
        returned[b] += 1.0;
        return_sum[b] += std::max(1.0, gap);
    } else {
        expired_beyond_h += 1.0;
    }
}

void EverglintJointPolicy::ReturnHistogram::observe_expired() {
    expired_beyond_h += 1.0;
}

std::pair<double, double> EverglintJointPolicy::ReturnHistogram::predict(
    double idle_age,
    double horizon,
    double prior_p,
    double prior_w
) const noexcept {
    const double age = std::max(0.0, idle_age);
    if (age >= horizon) {
        return {std::min(0.05, prior_p * 0.1), horizon};
    }
    const std::size_t start_b = age <= 0.0 ? 0 : bucket_for(age);
    double ret_ge = 0.0;
    double sum_ge = 0.0;
    for (std::size_t b = start_b; b < kNumBuckets; ++b) {
        ret_ge += returned[b];
        sum_ge += return_sum[b];
    }
    const double denom = ret_ge + expired_beyond_h + prior_w;
    const double p = denom > 0.0 ? std::clamp((ret_ge + prior_w * prior_p) / denom, 0.0, 1.0)
                                 : prior_p;
    double tau = std::max(1.0, 0.25 * horizon);
    if (ret_ge >= 1.0) {
        const double mean_total_gap = sum_ge / ret_ge;
        tau = std::max(1.0, mean_total_gap - age);
    }
    return {p, std::min(tau, horizon)};
}

double EverglintJointPolicy::ReturnHistogram::predict_survival(
    double idle_age,
    double horizon,
    double residence_scale,
    double prior_p,
    double prior_w
) const noexcept {
    if (residence_scale <= 0.0) {
        return 1.0;
    }
    const double age = std::max(0.0, idle_age);
    if (age >= horizon) {
        return std::exp(-horizon / residence_scale);
    }
    const std::size_t start_b = age <= 0.0 ? 0 : bucket_for(age);
    double ret_ge = 0.0;
    double surv_sum = 0.0;
    for (std::size_t b = start_b; b < kNumBuckets; ++b) {
        if (returned[b] <= 0.0) {
            continue;
        }
        const double mean_b = return_sum[b] / returned[b];
        const double rem = std::max(0.0, mean_b - age);
        ret_ge += returned[b];
        surv_sum += returned[b] * std::exp(-rem / residence_scale);
    }
    const double prior_tau = std::max(1.0, 0.1 * horizon);
    const double prior_surv = std::exp(-prior_tau / residence_scale);
    const double w = std::max(1.0, prior_w * prior_p);
    return std::clamp((surv_sum + w * prior_surv) / (ret_ge + w), 0.0, 1.0);
}

EverglintJointPolicy::Mode EverglintJointPolicy::parse_mode(const std::string& mode) {
    if (mode == "everglint") {
        return Mode::Everglint;
    }
    if (mode == "gandalf_tlc") {
        return Mode::GandalfTlc;
    }
    if (mode == "gandalf_slc") {
        return Mode::GandalfSlc;
    }
    if (mode == "gandalf_wear_share") {
        return Mode::GandalfWearShare;
    }
    throw std::invalid_argument(
        "everglint mode must be everglint, gandalf_tlc, gandalf_slc, or gandalf_wear_share"
    );
}

std::uint64_t EverglintJointPolicy::stable_hash(
    std::uint64_t value,
    std::uint64_t salt
) noexcept {
    std::uint64_t result = value + 0x9E3779B97F4A7C15ULL + salt;
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9ULL;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EBULL;
    return result ^ (result >> 31);
}

EverglintJointPolicy::EverglintJointPolicy(EverglintJointPolicyConfig config)
    : config_(std::move(config)), mode_(parse_mode(config_.mode)) {
    if (!std::isfinite(config_.wear_lambda) || config_.wear_lambda < 0.0) {
        throw std::invalid_argument("wear_lambda must be finite and nonnegative");
    }
    if (!std::isfinite(config_.wear_eta) || config_.wear_eta < 0.0) {
        throw std::invalid_argument("wear_eta must be finite and nonnegative");
    }
    if (config_.slc_pe_limit <= 0.0 || config_.tlc_pe_limit <= 0.0) {
        throw std::invalid_argument("slc_pe_limit and tlc_pe_limit must be positive");
    }
    if (config_.slc_waf_est <= 0.0 || config_.tlc_waf_est <= 0.0) {
        throw std::invalid_argument("slc_waf_est and tlc_waf_est must be positive");
    }
    if (config_.horizon_requests == 0 || config_.chunk_blocks == 0 ||
        config_.candidate_limit == 0 || config_.victim_limit == 0) {
        throw std::invalid_argument("horizon, chunk_blocks, candidate_limit, victim_limit must be positive");
    }
    if (config_.logical_fill_fraction <= 0.0 || config_.logical_fill_fraction > 1.0) {
        throw std::invalid_argument("logical_fill_fraction must be in (0, 1]");
    }
    for (auto& hist : histograms_) {
        hist.init(static_cast<double>(config_.horizon_requests));
    }
}

std::size_t EverglintJointPolicy::session_stratum(AffinityId sid) const noexcept {
    const auto it = sessions_.find(sid);
    if (it == sessions_.end() || it->second.req_count < 2 || it->second.ema_gap <= 0.0) {
        return 0;
    }
    const double g = it->second.ema_gap;
    if (g <= 32.0) {
        return 1;  // Fast interactive / agent tool-call loop
    }
    if (g <= 256.0) {
        return 2;  // Medium multi-turn cadence
    }
    return 3;      // Slow / long-think cadence
}

void EverglintJointPolicy::bind_tree(const RadixTree& tree) {
    tree_ = &tree;
}

bool EverglintJointPolicy::admit_storage_hit(
    const AccessContext&,
    const Node&,
    const RadixTree&
) const {
    return config_.admit_storage_hits;
}

void EverglintJointPolicy::on_request_begin(
    const RequestContext& request,
    const StorageView&
) {
    ++current_req_seq_;
    current_affinity_ = request.affinity_id == 0 ? request.request_id : request.affinity_id;
    current_req_blocks_ = std::max<double>(1.0, static_cast<double>(request.ordered_hashes.size));

    // Expire sessions whose idle age has crossed horizon H (right-censoring safe).
    const std::uint64_t h = config_.horizon_requests;
    while (!session_recency_.empty()) {
        const auto [last_seq, sid] = *session_recency_.begin();
        if (current_req_seq_ <= last_seq + h) {
            break;
        }
        session_recency_.erase(session_recency_.begin());
        auto it = sessions_.find(sid);
        if (it != sessions_.end() && !it->second.expired_counted) {
            it->second.expired_counted = true;
            histograms_[session_stratum(sid)].observe_expired();
        }
    }

    auto [it, inserted] = sessions_.try_emplace(current_affinity_);
    SessionState& st = it->second;
    if (!inserted) {
        const std::size_t prev_stratum = session_stratum(current_affinity_);
        const double gap = static_cast<double>(current_req_seq_ - st.last_req_seq);
        if (!st.expired_counted) {
            session_recency_.erase({st.last_req_seq, current_affinity_});
            histograms_[prev_stratum].observe_return(gap, static_cast<double>(h));
        }
        if (st.ema_gap <= 0.0) {
            st.ema_gap = gap;
        } else {
            st.ema_gap = 0.65 * st.ema_gap + 0.35 * gap;
        }
        st.ema_context_blocks = 0.7 * st.ema_context_blocks + 0.3 * current_req_blocks_;
        if (current_req_blocks_ > st.ema_context_blocks) {
            st.ema_context_blocks = current_req_blocks_;
        }
    } else {
        st.ema_context_blocks = current_req_blocks_;
    }
    st.last_req_seq = current_req_seq_;
    ++st.req_count;
    st.expired_counted = false;
    session_recency_.insert({current_req_seq_, current_affinity_});
}

void EverglintJointPolicy::on_request_end(
    const RequestContext&,
    const StorageView&,
    std::uint64_t,
    std::uint64_t
) {}

void EverglintJointPolicy::on_node_created(NodeId node_id, const RadixTree& tree) {
    NodeMeta meta;
    const auto parent = tree.parent(node_id);
    if (parent.has_value()) {
        const auto pit = nodes_.find(*parent);
        meta.depth = pit != nodes_.end() ? pit->second.depth + 1 : 1;
    } else {
        meta.depth = 1;
    }
    meta.last_req_seq = current_req_seq_;
    meta.session_id = current_affinity_;
    meta.expected_context_blocks = std::max(current_req_blocks_, static_cast<double>(meta.depth));
    nodes_[node_id] = meta;
}

void EverglintJointPolicy::on_node_pruned(
    NodeId node_id,
    std::optional<NodeId>,
    const RadixTree&
) {
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) {
        remove_from_all_sets(node_id, it->second);
        nodes_.erase(it);
    }
}

void EverglintJointPolicy::remove_from_all_sets(NodeId id, NodeMeta& meta) {
    const RecencyKey key{meta.indexed_key_seq, id};
    if (meta.in_dram_clean) {
        dram_clean_set_.erase(key);
        meta.in_dram_clean = false;
    }
    if (meta.in_dram_leaf) {
        dram_leaf_set_.erase(key);
        meta.in_dram_leaf = false;
    }
    if (meta.in_dram_root) {
        dram_root_set_.erase(key);
        meta.in_dram_root = false;
    }
    for (std::size_t t = 0; t < 2; ++t) {
        if (meta.in_ssd_leaf[t]) {
            ssd_leaf_set_[t].erase(key);
            meta.in_ssd_leaf[t] = false;
        }
    }
}

void EverglintJointPolicy::refresh_node_sets(NodeId id, const RadixTree& tree) {
    auto it = nodes_.find(id);
    if (it == nodes_.end() || !tree.contains(id)) {
        return;
    }
    NodeMeta& meta = it->second;
    remove_from_all_sets(id, meta);

    const Node& node = tree.node(id);
    const bool cached_now = node.in_memory || node.on_storage;
    if (cached_now != meta.was_cached) {
        meta.was_cached = cached_now;
        const auto parent = tree.parent(id);
        if (parent.has_value()) {
            auto pit = nodes_.find(*parent);
            if (pit != nodes_.end()) {
                if (cached_now) {
                    ++pit->second.cached_child_count;
                } else if (pit->second.cached_child_count > 0) {
                    --pit->second.cached_child_count;
                }
                refresh_node_sets(*parent, tree);
            }
        }
    }

    if (!cached_now) {
        return;
    }

    meta.indexed_key_seq = meta.last_req_seq;
    const RecencyKey key{meta.indexed_key_seq, id};

    if (node.in_memory && node.on_storage) {
        dram_clean_set_.insert(key);
        meta.in_dram_clean = true;
    } else if (node.in_memory && !node.on_storage) {
        if (meta.cached_child_count == 0) {
            dram_leaf_set_.insert(key);
            meta.in_dram_leaf = true;
        }
        const auto parent = tree.parent(id);
        const bool parent_on_ssd = !parent.has_value() ||
                                   (tree.contains(*parent) && tree.node(*parent).on_storage);
        if (parent_on_ssd) {
            dram_root_set_.insert(key);
            meta.in_dram_root = true;
        }
    } else if (!node.in_memory && node.on_storage) {
        if (meta.cached_child_count == 0) {
            const std::size_t t = storage_tier_index(node.storage_tier);
            ssd_leaf_set_[t].insert(key);
            meta.in_ssd_leaf[t] = true;
        }
    }
}

void EverglintJointPolicy::refresh_children_root_membership(
    NodeId parent_id,
    const RadixTree& tree
) {
    if (!tree.contains(parent_id)) {
        return;
    }
    std::vector<NodeId> children;
    tree.children(parent_id, children);
    for (NodeId child : children) {
        refresh_node_sets(child, tree);
    }
}

void EverglintJointPolicy::on_memory_commit(
    const MemoryMutation& mutation,
    const RadixTree& tree
) {
    auto it = nodes_.find(mutation.node_id);
    if (it == nodes_.end()) {
        return;
    }
    NodeMeta& meta = it->second;
    if (mutation.kind == MemoryMutationKind::Inserted ||
        mutation.kind == MemoryMutationKind::Accessed) {
        if (meta.reuse_count == 0 || meta.last_req_seq != current_req_seq_) {
            ++meta.reuse_count;
        }
        meta.last_req_seq = current_req_seq_;
        meta.session_id = current_affinity_;
        meta.expected_context_blocks = std::max(
            0.7 * meta.expected_context_blocks + 0.3 * current_req_blocks_,
            static_cast<double>(meta.depth)
        );
    }
    refresh_node_sets(mutation.node_id, tree);
}

void EverglintJointPolicy::on_storage_commit(
    const StorageMutation& mutation,
    const StorageView& storage_after_commit
) {
    const RadixTree& tree = storage_after_commit.tree();
    switch (mutation.kind) {
        case StorageMutationKind::DumpWriteCommitted: {
            const std::size_t t = storage_tier_index(mutation.placement.tier);
            const double waf = effective_waf(mutation.placement.tier, storage_after_commit);
            program_bytes_[t] += mutation.bytes;
            est_nand_bytes_[t] += waf * static_cast<double>(mutation.bytes);
            if (!mutation.nodes.empty()) {
                const auto nit = nodes_.find(mutation.nodes[0]);
                if (nit != nodes_.end() && nit->second.session_id != 0) {
                    auto sit = sessions_.find(nit->second.session_id);
                    if (sit != sessions_.end() && !sit->second.preferred_tier.has_value()) {
                        sit->second.preferred_tier = mutation.placement.tier;
                    }
                }
            }
            for (NodeId id : mutation.nodes) {
                auto it = nodes_.find(id);
                if (it != nodes_.end()) {
                    it->second.write_req_seq = current_req_seq_;
                }
                refresh_node_sets(id, tree);
                refresh_children_root_membership(id, tree);
            }
            break;
        }
        case StorageMutationKind::CapacityTrimCommitted: {
            const std::size_t t = storage_tier_index(mutation.source_tier);
            const double alpha = config_.residence_ema_alpha;
            for (NodeId id : mutation.nodes) {
                auto it = nodes_.find(id);
                if (it != nodes_.end()) {
                    const double res_age = std::max(
                        1.0,
                        static_cast<double>(current_req_seq_ - it->second.write_req_seq)
                    );
                    if (eviction_samples_[t] == 0) {
                        residence_ema_[t] = res_age;
                    } else {
                        residence_ema_[t] = (1.0 - alpha) * residence_ema_[t] + alpha * res_age;
                    }
                    ++eviction_samples_[t];
                    if (global_eviction_samples_ == 0) {
                        global_residence_ema_ = res_age;
                    } else {
                        global_residence_ema_ =
                            (1.0 - alpha) * global_residence_ema_ + alpha * res_age;
                    }
                    ++global_eviction_samples_;
                }
                refresh_node_sets(id, tree);
                refresh_children_root_membership(id, tree);
            }
            break;
        }
        case StorageMutationKind::IdleTrimCommitted: {
            for (NodeId id : mutation.nodes) {
                refresh_node_sets(id, tree);
                refresh_children_root_membership(id, tree);
            }
            break;
        }
        case StorageMutationKind::StorageAccessCommitted: {
            for (NodeId id : mutation.nodes) {
                auto it = nodes_.find(id);
                if (it != nodes_.end()) {
                    if (it->second.reuse_count == 0 || it->second.last_req_seq != current_req_seq_) {
                        ++it->second.reuse_count;
                    }
                    it->second.last_req_seq = current_req_seq_;
                    it->second.session_id = current_affinity_;
                    it->second.expected_context_blocks = std::max(
                        0.7 * it->second.expected_context_blocks + 0.3 * current_req_blocks_,
                        static_cast<double>(it->second.depth)
                    );
                    if (it->second.in_ssd_leaf[0] || it->second.in_ssd_leaf[1] ||
                        it->second.in_dram_clean) {
                        refresh_node_sets(id, tree);
                    }
                }
            }
            break;
        }
        case StorageMutationKind::NodePruned:
        case StorageMutationKind::RelocationReadCommitted:
        case StorageMutationKind::RelocationWriteCommitted:
        case StorageMutationKind::RelocationSourceTrimCommitted:
            break;
    }
}

double EverglintJointPolicy::region_budget_bytes(
    StorageTier tier,
    const StorageView& storage
) const noexcept {
    const std::uint64_t logical =
        storage.storage().tier(tier).capacity_blocks() * storage.block_size_bytes();
    const std::uint64_t phys = tier == StorageTier::Slc
                                   ? (config_.slc_physical_bytes ? config_.slc_physical_bytes : logical)
                                   : (config_.tlc_physical_bytes ? config_.tlc_physical_bytes : logical);
    const double pe = tier == StorageTier::Slc ? config_.slc_pe_limit : config_.tlc_pe_limit;
    return static_cast<double>(phys) * pe;
}

double EverglintJointPolicy::effective_waf(
    StorageTier tier,
    const StorageView& storage
) const noexcept {
    const double base_waf = tier == StorageTier::Slc ? config_.slc_waf_est : config_.tlc_waf_est;
    const auto& ts = storage.storage().tier(tier);
    const std::uint64_t cap = ts.capacity_blocks();
    if (cap == 0) {
        return base_waf;
    }
    const double u_log = static_cast<double>(ts.used_blocks()) / static_cast<double>(cap);
    const std::uint64_t logical_bytes = cap * storage.block_size_bytes();
    const std::uint64_t phys_bytes =
        tier == StorageTier::Slc
            ? (config_.slc_physical_bytes ? config_.slc_physical_bytes : logical_bytes)
            : (config_.tlc_physical_bytes ? config_.tlc_physical_bytes : logical_bytes);
    const double fill_ratio =
        phys_bytes > 0 ? std::clamp(static_cast<double>(logical_bytes) / static_cast<double>(phys_bytes), 0.5, 0.98)
                       : 0.90;
    // Macroscopic steady-state greedy-GC WAF proxy: scales inversely with effective free fraction.
    const double eff_free = std::max(0.08, 1.0 - u_log * fill_ratio);
    const double gc_waf = std::max(1.0, 0.5 / eff_free);
    return base_waf * gc_waf;
}

double EverglintJointPolicy::region_wear(
    StorageTier tier,
    const StorageView& storage
) const noexcept {
    const std::size_t idx = storage_tier_index(tier);
    const double budget = region_budget_bytes(tier, storage);
    return budget > 0.0 ? est_nand_bytes_[idx] / budget : 0.0;
}

double EverglintJointPolicy::wear_potential(double rho_s, double rho_t) const noexcept {
    return std::max(rho_s, rho_t) + config_.wear_eta * (rho_s + rho_t);
}

double EverglintJointPolicy::marginal_wear(
    StorageTier tier,
    std::uint64_t write_bytes,
    const StorageView& storage
) const noexcept {
    const double rho_s = region_wear(StorageTier::Slc, storage);
    const double rho_t = region_wear(StorageTier::Tlc, storage);
    const double waf = effective_waf(tier, storage);
    const double delta = (waf * static_cast<double>(write_bytes)) / region_budget_bytes(tier, storage);
    if (config_.wear_eta <= 0.0) {
        // Exact hard minimax for NoEta ablation.
        const double next = tier == StorageTier::Slc
                                ? wear_potential(rho_s + delta, rho_t)
                                : wear_potential(rho_s, rho_t + delta);
        return next - wear_potential(rho_s, rho_t);
    }
    // Scale-invariant smooth minimax gradient: avoids 11x step-kink oscillation at rho_S == rho_T
    // while smoothly steering writes away from whichever tier pulls ahead in normalized wear.
    const double total = rho_s + rho_t;
    const double diff = (tier == StorageTier::Slc) ? (rho_s - rho_t) : (rho_t - rho_s);
    const double z = total > 1e-15 ? std::clamp(6.0 * diff / total, -12.0, 12.0) : 0.0;
    const double minimax_weight = 1.0 / (1.0 + std::exp(-z));
    return (minimax_weight + config_.wear_eta) * delta;
}

double EverglintJointPolicy::uniform_marginal_wear(
    std::uint64_t write_bytes,
    const StorageView& storage
) const noexcept {
    if (mode_ == Mode::GandalfTlc) {
        const double delta =
            (config_.tlc_waf_est * static_cast<double>(write_bytes)) /
            region_budget_bytes(StorageTier::Tlc, storage);
        return (1.0 + config_.wear_eta) * delta;
    }
    if (mode_ == Mode::GandalfSlc) {
        const double delta =
            (config_.slc_waf_est * static_cast<double>(write_bytes)) /
            region_budget_bytes(StorageTier::Slc, storage);
        return (1.0 + config_.wear_eta) * delta;
    }
    // Gandalf-WearShare treats the dual-region SSD as one combined endurance pool B_S + B_T.
    const double total_budget =
        region_budget_bytes(StorageTier::Slc, storage) +
        region_budget_bytes(StorageTier::Tlc, storage);
    return (1.0 + 2.0 * config_.wear_eta) * (static_cast<double>(write_bytes) / total_budget);
}

double EverglintJointPolicy::survival_prob(
    const NodeMeta& meta,
    StorageTier tier
) const noexcept {
    if (!config_.use_survival) {
        return 1.0;
    }
    const std::size_t idx = storage_tier_index(tier);
    if (eviction_samples_[idx] == 0 || residence_ema_[idx] <= 0.0) {
        return 1.0;
    }
    const double idle_age =
        current_req_seq_ > meta.last_req_seq
            ? static_cast<double>(current_req_seq_ - meta.last_req_seq)
            : 0.0;
    const std::size_t hist_idx = session_stratum(meta.session_id);
    return histograms_[hist_idx].predict_survival(
        idle_age,
        static_cast<double>(config_.horizon_requests),
        residence_ema_[idx],
        config_.prior_reuse_prob,
        config_.prior_weight
    );
}

double EverglintJointPolicy::uniform_survival_prob(const NodeMeta& meta) const noexcept {
    if (!config_.use_survival) {
        return 1.0;
    }
    if (global_eviction_samples_ == 0 || global_residence_ema_ <= 0.0) {
        return 1.0;
    }
    const double idle_age =
        current_req_seq_ > meta.last_req_seq
            ? static_cast<double>(current_req_seq_ - meta.last_req_seq)
            : 0.0;
    const std::size_t hist_idx = session_stratum(meta.session_id);
    return histograms_[hist_idx].predict_survival(
        idle_age,
        static_cast<double>(config_.horizon_requests),
        global_residence_ema_,
        config_.prior_reuse_prob,
        config_.prior_weight
    );
}

std::pair<double, double> EverglintJointPolicy::predict_node(
    const NodeMeta& meta
) const noexcept {
    const double idle_age =
        current_req_seq_ > meta.last_req_seq
            ? static_cast<double>(current_req_seq_ - meta.last_req_seq)
            : 0.0;
    const std::size_t hist_idx = session_stratum(meta.session_id);
    return histograms_[hist_idx].predict(
        idle_age,
        static_cast<double>(config_.horizon_requests),
        config_.prior_reuse_prob,
        config_.prior_weight
    );
}

double EverglintJointPolicy::block_recompute_cost(const NodeMeta& meta) const noexcept {
    if (!config_.use_prefix_cost) {
        return 1.0;
    }
    double ctx = meta.expected_context_blocks;
    const auto sit = sessions_.find(meta.session_id);
    if (sit != sessions_.end()) {
        ctx = std::max(ctx, sit->second.ema_context_blocks);
    }
    return std::max(ctx, static_cast<double>(meta.depth));
}

std::uint64_t EverglintJointPolicy::tier_capacity_limit(
    StorageTier tier,
    const StorageView& storage
) const noexcept {
    const std::uint64_t raw = storage.storage().tier(tier).capacity_blocks();
    return std::max<std::uint64_t>(
        1,
        static_cast<std::uint64_t>(static_cast<double>(raw) * config_.logical_fill_fraction)
    );
}

std::uint32_t EverglintJointPolicy::stream_for(
    StorageTier tier,
    AffinityId affinity_id,
    NodeId endpoint,
    const StorageView& storage
) const noexcept {
    const std::uint32_t count = storage.storage().tier(tier).stream_count();
    if (count <= 1) {
        return 0;
    }
    const std::uint64_t key = affinity_id == 0 ? endpoint : affinity_id;
    return static_cast<std::uint32_t>(stable_hash(key, 29) % count);
}

bool EverglintJointPolicy::build_ssd_victims(
    StorageTier tier,
    std::uint64_t need_blocks,
    NodeSpan protected_nodes,
    const std::vector<NodeId>& moving_nodes,
    const StorageView& storage,
    std::vector<NodeId>& out_victims,
    double& out_replacement_loss
) const {
    out_victims.clear();
    out_replacement_loss = 0.0;
    if (need_blocks == 0) {
        return true;
    }

    const RadixTree& tree = storage.tree();
    std::unordered_set<NodeId> forbidden(protected_nodes.begin(), protected_nodes.end());
    forbidden.insert(moving_nodes.begin(), moving_nodes.end());

    struct VictimChain {
        std::vector<NodeId> nodes_bottom_up;
        double loss_per_block = 0.0;
        double total_loss = 0.0;
        std::uint64_t recency = 0;
    };

    const auto& leaf_set = ssd_leaf_set_[storage_tier_index(tier)];
    std::vector<VictimChain> candidates;
    candidates.reserve(std::min<std::size_t>(leaf_set.size(), config_.victim_limit));

    std::size_t scanned = 0;
    for (const auto& [recency, leaf_id] : leaf_set) {
        if (scanned >= config_.victim_limit && candidates.size() >= need_blocks) {
            break;
        }
        ++scanned;
        if (forbidden.count(leaf_id) || !tree.contains(leaf_id)) {
            continue;
        }

        // Walk up a single-child same-tier SSD chain (bounded by max(chunk_blocks, need_blocks)).
        VictimChain chain;
        chain.recency = recency;
        const std::size_t max_len = static_cast<std::size_t>(
            std::max(config_.chunk_blocks, need_blocks)
        );
        std::optional<NodeId> cur = leaf_id;
        double chain_loss = 0.0;
        while (cur.has_value() && chain.nodes_bottom_up.size() < max_len) {
            if (forbidden.count(*cur) || !tree.contains(*cur)) {
                break;
            }
            const Node& node = tree.node(*cur);
            if (!node.on_storage || node.storage_tier != tier || node.in_memory) {
                break;
            }
            const auto mit = nodes_.find(*cur);
            if (mit == nodes_.end()) {
                break;
            }
            if (cur != leaf_id && mit->second.cached_child_count > 1) {
                break;
            }
            const auto [p_e, tau_e] = predict_node(mit->second);
            static_cast<void>(tau_e);
            const double q_e = (mode_ == Mode::Everglint)
                                   ? survival_prob(mit->second, tier)
                                   : uniform_survival_prob(mit->second);
            const double c_e = block_recompute_cost(mit->second);
            chain_loss += p_e * q_e * c_e;
            chain.nodes_bottom_up.push_back(*cur);
            cur = tree.parent(*cur);
        }
        if (chain.nodes_bottom_up.empty()) {
            continue;
        }
        chain.total_loss = chain_loss;
        chain.loss_per_block = chain_loss / static_cast<double>(chain.nodes_bottom_up.size());
        candidates.push_back(std::move(chain));
    }

    if (config_.use_value_victim) {
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const VictimChain& a, const VictimChain& b) {
                if (a.loss_per_block != b.loss_per_block) {
                    return a.loss_per_block < b.loss_per_block;
                }
                return a.recency < b.recency;
            }
        );
    }

    // Trim whole selected victim chains (up to chunk_blocks) so contiguous session blocks
    // invalidate cleanly together and minimize flash erase-block fragmentation (WAF).
    std::unordered_set<NodeId> selected;
    std::uint64_t freed = 0;
    for (const auto& chain : candidates) {
        if (freed >= need_blocks) {
            break;
        }
        for (NodeId id : chain.nodes_bottom_up) {
            if (!selected.insert(id).second) {
                break;
            }
            const auto& meta = nodes_.at(id);
            const auto [p_e, tau_e] = predict_node(meta);
            static_cast<void>(tau_e);
            const double q_e = (mode_ == Mode::Everglint)
                                   ? survival_prob(meta, tier)
                                   : uniform_survival_prob(meta);
            const double c_e = block_recompute_cost(meta);
            out_replacement_loss += p_e * q_e * c_e;
            out_victims.push_back(id);
            ++freed;
        }
    }

    if (freed < need_blocks) {
        // Scan remaining leaves beyond victim_limit if capacity is severely constrained.
        std::size_t idx = 0;
        for (const auto& [recency, leaf_id] : leaf_set) {
            ++idx;
            if (idx <= config_.victim_limit) {
                continue;
            }
            if (freed >= need_blocks) {
                break;
            }
            if (forbidden.count(leaf_id) || selected.count(leaf_id) || !tree.contains(leaf_id)) {
                continue;
            }
            std::optional<NodeId> cur = leaf_id;
            while (cur.has_value() && freed < need_blocks) {
                if (forbidden.count(*cur) || selected.count(*cur) || !tree.contains(*cur)) {
                    break;
                }
                const Node& node = tree.node(*cur);
                if (!node.on_storage || node.storage_tier != tier || node.in_memory) {
                    break;
                }
                const auto mit = nodes_.find(*cur);
                if (mit == nodes_.end() || (cur != leaf_id && mit->second.cached_child_count > 1)) {
                    break;
                }
                selected.insert(*cur);
                const auto [p_e, tau_e] = predict_node(mit->second);
                static_cast<void>(tau_e);
                const double q_e = (mode_ == Mode::Everglint)
                                       ? survival_prob(mit->second, tier)
                                       : uniform_survival_prob(mit->second);
                out_replacement_loss += p_e * q_e * block_recompute_cost(mit->second);
                out_victims.push_back(*cur);
                ++freed;
                cur = tree.parent(*cur);
            }
        }
    }

    return freed >= need_blocks;
}

JointEvictionPlan EverglintJointPolicy::plan_eviction(
    const AccessContext& context,
    const StorageView& storage
) const {
    const RadixTree& tree = storage.tree();
    std::unordered_set<NodeId> protected_set(
        context.request.protected_prefix.begin(),
        context.request.protected_prefix.end()
    );
    protected_set.insert(context.node_id);

    // 1. Clean DRAM eviction (already has SSD copy -> 0 write, 0 recompute loss).
    if (!dram_clean_set_.empty()) {
        JointEvictionPlan clean;
        clean.kind = JointActionKind::CleanEvict;
        for (const auto& [recency, id] : dram_clean_set_) {
            if (protected_set.count(id) || !tree.contains(id)) {
                continue;
            }
            clean.memory_nodes.push_back(id);
            if (clean.memory_nodes.size() >= config_.chunk_blocks) {
                break;
            }
        }
        if (!clean.memory_nodes.empty()) {
            return clean;
        }
    }

    double best_loss_per_block = std::numeric_limits<double>::infinity();
    std::optional<JointEvictionPlan> best_plan;

    // 2. Evaluate legal DRAM-leaf Drop candidates.
    if (config_.allow_drop) {
        std::size_t scanned = 0;
        for (const auto& [recency, leaf_id] : dram_leaf_set_) {
            if (scanned >= config_.candidate_limit) {
                break;
            }
            if (protected_set.count(leaf_id) || !tree.contains(leaf_id)) {
                continue;
            }
            ++scanned;

            // Walk up a single-child DRAM-only chain up to chunk_blocks.
            std::vector<NodeId> chunk_bottom_up;
            std::optional<NodeId> cur = leaf_id;
            double total_loss = 0.0;
            while (cur.has_value() && chunk_bottom_up.size() < config_.chunk_blocks) {
                if (protected_set.count(*cur) || !tree.contains(*cur)) {
                    break;
                }
                const Node& node = tree.node(*cur);
                if (!node.in_memory || node.on_storage) {
                    break;
                }
                const auto mit = nodes_.find(*cur);
                if (mit == nodes_.end()) {
                    break;
                }
                if (cur != leaf_id && mit->second.cached_child_count > 1) {
                    break;
                }
                const auto [p_i, tau_i] = predict_node(mit->second);
                const double c_i = block_recompute_cost(mit->second);
                total_loss += p_i * c_i;
                chunk_bottom_up.push_back(*cur);
                cur = tree.parent(*cur);
            }
            if (chunk_bottom_up.empty()) {
                continue;
            }
            const double loss_per_block =
                total_loss / static_cast<double>(chunk_bottom_up.size());
            if (loss_per_block < best_loss_per_block) {
                best_loss_per_block = loss_per_block;
                JointEvictionPlan plan;
                plan.kind = JointActionKind::Drop;
                plan.memory_nodes.assign(chunk_bottom_up.rbegin(), chunk_bottom_up.rend());
                best_plan = std::move(plan);
            }
        }
    }

    // 3. Evaluate legal DRAM root-side Move-to-SLC and Move-to-TLC candidates.
    const std::uint64_t block_size = storage.block_size_bytes();
    const double slc_b = region_budget_bytes(StorageTier::Slc, storage);
    const double tlc_b = region_budget_bytes(StorageTier::Tlc, storage);
    const double slc_target_share = slc_b / std::max(1.0, slc_b + tlc_b);

    std::size_t scanned_roots = 0;
    for (const auto& [recency, root_id] : dram_root_set_) {
        if (scanned_roots >= config_.candidate_limit) {
            break;
        }
        if (protected_set.count(root_id) || !tree.contains(root_id)) {
            continue;
        }
        ++scanned_roots;

        // Walk top-down along a single-child DRAM-only chain up to chunk_blocks.
        std::vector<NodeId> chunk_top_down;
        std::optional<NodeId> cur = root_id;
        std::vector<NodeId> children;
        while (cur.has_value() && chunk_top_down.size() < config_.chunk_blocks) {
            if (protected_set.count(*cur) || !tree.contains(*cur)) {
                break;
            }
            const Node& node = tree.node(*cur);
            if (!node.in_memory || node.on_storage) {
                break;
            }
            chunk_top_down.push_back(*cur);
            if (tree.child_count(*cur) != 1) {
                break;
            }
            children.clear();
            tree.children(*cur, children);
            if (children.size() != 1) {
                break;
            }
            cur = children[0];
        }
        if (chunk_top_down.empty()) {
            continue;
        }

        const std::uint64_t m = chunk_top_down.size();
        const std::uint64_t write_bytes = m * block_size;
        const auto& root_meta = nodes_.at(root_id);

        // Determine candidate destination tiers according to mode.
        // Session-coherent & Prefix-Monotonic placement for Everglint: keep chunks of the same
        // session co-located on one tier (preventing tier-inversion pinning and flash-block
        // lifetime mixing), falling back if the preferred tier cannot free space.
        const auto parent_opt = tree.parent(root_id);
        const bool parent_on_ssd =
            parent_opt.has_value() && tree.contains(*parent_opt) &&
            tree.node(*parent_opt).on_storage;
        const bool parent_in_slc =
            parent_on_ssd && tree.node(*parent_opt).storage_tier == StorageTier::Slc;

        std::optional<StorageTier> session_tier;
        if (root_meta.session_id != 0) {
            const auto sit = sessions_.find(root_meta.session_id);
            if (sit != sessions_.end()) {
                session_tier = sit->second.preferred_tier;
            }
        }

        std::vector<StorageTier> candidate_tiers;
        bool stop_after_first_feasible = false;
        if (mode_ == Mode::Everglint) {
            if (parent_in_slc) {
                candidate_tiers = config_.allow_drop
                                      ? std::vector<StorageTier>{StorageTier::Slc}
                                      : std::vector<StorageTier>{StorageTier::Slc, StorageTier::Tlc};
                stop_after_first_feasible = true;
            } else if (session_tier.has_value()) {
                const StorageTier pref = *session_tier;
                const StorageTier alt =
                    pref == StorageTier::Slc ? StorageTier::Tlc : StorageTier::Slc;
                candidate_tiers = {pref, alt};
                stop_after_first_feasible = true;
            } else {
                candidate_tiers = {StorageTier::Slc, StorageTier::Tlc};
            }
        } else if (mode_ == Mode::GandalfTlc) {
            candidate_tiers = {StorageTier::Tlc};
        } else if (mode_ == Mode::GandalfSlc) {
            candidate_tiers = {StorageTier::Slc};
        } else {
            // Gandalf-WearShare: pick tier by endurance-proportional affinity hash.
            const std::uint64_t key =
                root_meta.session_id == 0 ? root_id : root_meta.session_id;
            constexpr std::uint64_t mask = (1ULL << 53) - 1;
            const double draw =
                static_cast<double>(stable_hash(key, 47) & mask) /
                static_cast<double>(1ULL << 53);
            const StorageTier primary =
                draw < slc_target_share ? StorageTier::Slc : StorageTier::Tlc;
            const StorageTier fallback =
                primary == StorageTier::Slc ? StorageTier::Tlc : StorageTier::Slc;
            candidate_tiers = {primary, fallback};
        }

        for (StorageTier tier : candidate_tiers) {
            const std::uint64_t limit = tier_capacity_limit(tier, storage);
            if (m > limit) {
                continue;
            }
            const std::uint64_t used = storage.storage().tier(tier).used_blocks();
            const std::uint64_t need = used + m > limit ? (used + m - limit) : 0;

            std::vector<NodeId> victims;
            double replacement_loss = 0.0;
            if (!build_ssd_victims(
                    tier,
                    need,
                    context.request.protected_prefix,
                    chunk_top_down,
                    storage,
                    victims,
                    replacement_loss
                )) {
                continue;
            }

            double early_evict_loss = 0.0;
            for (NodeId id : chunk_top_down) {
                const auto& meta = nodes_.at(id);
                const auto [p_i, tau_i] = predict_node(meta);
                static_cast<void>(tau_i);
                const double q_ik = (mode_ == Mode::Everglint)
                                        ? survival_prob(meta, tier)
                                        : uniform_survival_prob(meta);
                const double c_i = block_recompute_cost(meta);
                early_evict_loss += p_i * (1.0 - q_ik) * c_i;
            }

            const double wear_cost =
                config_.wear_lambda *
                ((mode_ == Mode::Everglint) ? marginal_wear(tier, write_bytes, storage)
                                            : uniform_marginal_wear(write_bytes, storage));
            const double total_loss = early_evict_loss + wear_cost + replacement_loss;
            const double loss_per_block = total_loss / static_cast<double>(m);

            if (loss_per_block < best_loss_per_block) {
                best_loss_per_block = loss_per_block;
                JointEvictionPlan plan;
                plan.kind = tier == StorageTier::Slc ? JointActionKind::MoveToSlc
                                                     : JointActionKind::MoveToTlc;
                plan.memory_nodes = chunk_top_down;
                plan.target_tier = tier;
                plan.stream_id = stream_for(tier, root_meta.session_id, root_id, storage);
                plan.ssd_trim_nodes = std::move(victims);
                best_plan = std::move(plan);
            }

            if (mode_ == Mode::GandalfWearShare || stop_after_first_feasible) {
                // Only use fallback tier if primary tier was infeasible.
                break;
            }
        }
    }

    if (best_plan.has_value()) {
        return *best_plan;
    }

    // Fallback: free an unprotected DRAM leaf (or deepest non-active leaf if all protected).
    for (const auto& [recency, leaf_id] : dram_leaf_set_) {
        if (leaf_id != context.node_id && !protected_set.count(leaf_id) && tree.contains(leaf_id)) {
            JointEvictionPlan plan;
            plan.kind = JointActionKind::Drop;
            plan.memory_nodes = {leaf_id};
            return plan;
        }
    }
    for (const auto& [recency, leaf_id] : dram_leaf_set_) {
        if (leaf_id != context.node_id && tree.contains(leaf_id)) {
            JointEvictionPlan plan;
            plan.kind = JointActionKind::Drop;
            plan.memory_nodes = {leaf_id};
            return plan;
        }
    }
    for (const auto& [recency, id] : dram_clean_set_) {
        if (id != context.node_id && tree.contains(id)) {
            JointEvictionPlan plan;
            plan.kind = JointActionKind::CleanEvict;
            plan.memory_nodes = {id};
            return plan;
        }
    }
    throw std::runtime_error("EverglintJointPolicy: no evictable DRAM node found");
}

std::vector<std::pair<StorageTier, NodeId>> EverglintJointPolicy::collect_proactive_trims(
    const RequestContext& request,
    const StorageView& storage
) const {
    std::vector<std::pair<StorageTier, NodeId>> result;
    if (config_.trim_gamma <= 0.0 || config_.wear_lambda <= 0.0) {
        return result;
    }
    // Check periodically (every 16 requests) for O(1) amortized overhead.
    if (current_req_seq_ % 16 != 0) {
        return result;
    }

    const RadixTree& tree = storage.tree();
    const std::uint64_t block_size = storage.block_size_bytes();
    std::unordered_set<NodeId> protected_set(
        request.protected_prefix.begin(),
        request.protected_prefix.end()
    );
    std::unordered_set<NodeId> selected;

    for (StorageTier tier : {StorageTier::Slc, StorageTier::Tlc}) {
        const auto& leaf_set = ssd_leaf_set_[storage_tier_index(tier)];
        if (leaf_set.empty()) {
            continue;
        }
        // Marginal GC holding cost is proportional to excess write amplification (WAF - 1):
        // when a tier has ample free space (WAF == 1), proactive trimming yields zero GC savings.
        const double excess_waf = std::max(0.0, effective_waf(tier, storage) - 1.0);
        if (excess_waf <= 0.0) {
            continue;
        }
        const double base_delta =
            (excess_waf * static_cast<double>(block_size)) / region_budget_bytes(tier, storage);
        const double trim_threshold =
            config_.trim_gamma * config_.wear_lambda * (0.5 + config_.wear_eta) * base_delta;
        if (trim_threshold <= 0.0) {
            continue;
        }

        std::size_t scanned = 0;
        for (const auto& [recency, leaf_id] : leaf_set) {
            if (scanned >= config_.victim_limit || result.size() >= 1024) {
                break;
            }
            ++scanned;
            if (protected_set.count(leaf_id) || selected.count(leaf_id) || !tree.contains(leaf_id)) {
                continue;
            }
            std::optional<NodeId> cur = leaf_id;
            std::size_t chain_len = 0;
            while (cur.has_value() && chain_len < 512 && result.size() < 1024) {
                if (protected_set.count(*cur) || selected.count(*cur) || !tree.contains(*cur)) {
                    break;
                }
                const Node& node = tree.node(*cur);
                if (!node.on_storage || node.storage_tier != tier || node.in_memory) {
                    break;
                }
                const auto mit = nodes_.find(*cur);
                if (mit == nodes_.end()) {
                    break;
                }
                if (cur != leaf_id && mit->second.cached_child_count > 1) {
                    break;
                }
                const auto [p_e, tau_e] = predict_node(mit->second);
                static_cast<void>(tau_e);
                const double q_e = (mode_ == Mode::Everglint)
                                       ? survival_prob(mit->second, tier)
                                       : uniform_survival_prob(mit->second);
                const double val = p_e * q_e * block_recompute_cost(mit->second);
                if (val >= trim_threshold) {
                    break;
                }
                selected.insert(*cur);
                result.emplace_back(tier, *cur);
                ++chain_len;
                cur = tree.parent(*cur);
            }
        }
    }
    return result;
}

StoragePolicyStats EverglintJointPolicy::stats(const StorageView&) const {
    StoragePolicyStats result;
    result.slc_program_bytes = program_bytes_[0];
    result.tlc_program_bytes = program_bytes_[1];
    return result;
}

}  // namespace dwpdsim
