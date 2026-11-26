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

void EverglintJointPolicy::ReturnHistogram::decay(double factor) noexcept {
    for (std::size_t i = 0; i < kNumBuckets; ++i) {
        returned[i] *= factor;
        return_sum[i] *= factor;
    }
    expired_beyond_h *= factor;
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
    double prior_w,
    bool jensen_exact
) const noexcept {
    if (residence_scale <= 0.0) {
        return 1.0;
    }
    const double age = std::max(0.0, idle_age);
    if (age >= horizon) {
        return std::exp(-horizon / residence_scale);
    }
    if (!jensen_exact) {
        // Plug-in conditional mean approximation exp(-E[T - a | T >= a] / L_k)
        const auto [p_unused, tau] = predict(age, horizon, prior_p, prior_w);
        static_cast<void>(p_unused);
        return std::clamp(std::exp(-tau / residence_scale), 0.0, 1.0);
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
    : config_(std::move(config)),
      mode_(parse_mode(config_.mode)),
      active_wear_lambda_(config_.wear_lambda) {
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
    if (!std::isfinite(config_.session_hysteresis) || config_.session_hysteresis < 0.0 ||
        config_.session_hysteresis >= 0.5) {
        throw std::invalid_argument("session_hysteresis must be in [0, 0.5)");
    }
    for (auto& hist : histograms_) {
        hist.init(static_cast<double>(config_.horizon_requests));
    }
}

std::size_t EverglintJointPolicy::session_stratum(AffinityId sid) const noexcept {
    if (!config_.use_cadence_strata) {
        return 0;
    }
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
    const AccessContext& access,
    const Node&,
    const RadixTree&
) const {
    if (config_.admit_storage_hits) {
        return true;
    }
    if (config_.promote_hit_threshold > 0) {
        const auto it = nodes_.find(access.node_id);
        if (it != nodes_.end() && (it->second.reuse_count + 1) >= config_.promote_hit_threshold) {
            return true;
        }
    }
    return false;
}

void EverglintJointPolicy::on_request_begin(
    const RequestContext& request,
    const StorageView& storage
) {
    // Update Little's Law per-request write-rate EMA from previous request.
    if (current_req_seq_ > 0) {
        constexpr double rate_alpha = 0.01;
        const double total_w = static_cast<double>(req_write_blocks_[0] + req_write_blocks_[1]);
        for (std::size_t t = 0; t < 2; ++t) {
            const double w = static_cast<double>(req_write_blocks_[t]);
            write_blocks_per_req_ema_[t] =
                (1.0 - rate_alpha) * write_blocks_per_req_ema_[t] + rate_alpha * w;
            req_write_blocks_[t] = 0;
        }
        global_write_blocks_per_req_ema_ =
            (1.0 - rate_alpha) * global_write_blocks_per_req_ema_ + rate_alpha * total_w;
    }

    // Optional closed-loop target-lifetime primal-dual lambda controller.
    if (config_.target_life_years > 0.0 && config_.dual_step_kappa > 0.0 &&
        config_.dual_window_ns > 0) {
        const double rho_s = region_wear(StorageTier::Slc, storage);
        const double rho_t = region_wear(StorageTier::Tlc, storage);
        if (!dual_window_start_ns_.has_value()) {
            dual_window_start_ns_ = request.timestamp_ns;
            dual_window_start_rho_ = {rho_s, rho_t};
        } else if (request.timestamp_ns >= *dual_window_start_ns_ + config_.dual_window_ns) {
            const double dt_ns = static_cast<double>(request.timestamp_ns - *dual_window_start_ns_);
            constexpr double year_ns = 365.0 * 86400.0 * 1.0e9;
            const double dt_years = dt_ns / year_ns;
            if (dt_years > 0.0) {
                const double d_rho_s = std::max(0.0, rho_s - dual_window_start_rho_[0]);
                const double d_rho_t = std::max(0.0, rho_t - dual_window_start_rho_[1]);
                const double obs_rate = std::max(d_rho_s, d_rho_t) / dt_years;
                constexpr double alpha_r = 0.5;
                dual_rate_ema_ = dual_rate_ema_.has_value()
                                     ? ((1.0 - alpha_r) * (*dual_rate_ema_) + alpha_r * obs_rate)
                                     : obs_rate;
                const double target_rate = 1.0 / config_.target_life_years;
                const double rel_err = std::clamp((*dual_rate_ema_) / target_rate - 1.0, -1.0, 1.0);
                active_wear_lambda_ = std::clamp(
                    active_wear_lambda_ * std::exp(config_.dual_step_kappa * rel_err),
                    1.0e9,
                    3.0e11
                );
                dual_trace_.push_back(EverglintDualSample{
                    request.timestamp_ns,
                    active_wear_lambda_,
                    obs_rate,
                    *dual_rate_ema_,
                    target_rate,
                    rho_s,
                    rho_t,
                });
            }
            dual_window_start_ns_ = request.timestamp_ns;
            dual_window_start_rho_ = {rho_s, rho_t};
        }
    }

    ++current_req_seq_;
    // Periodic exponential histogram aging (half-life = horizon_requests) so stale sessions
    // from prior epochs or non-stationary workload shifts decay cleanly instead of pinning p_i.
    if (current_req_seq_ % 256 == 0) {
        const double decay_factor =
            std::pow(0.5, 256.0 / static_cast<double>(config_.horizon_requests));
        for (auto& hist : histograms_) {
            hist.decay(decay_factor);
        }
    }
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
    auto session_class = [](AffinityId sid) noexcept -> std::size_t {
        const AffinityId base = sid >= 50000000ULL ? (sid - 50000000ULL) : sid;
        return base >= 100000ULL ? 1 : 0;  // 0 = Agent, 1 = Chat
    };
    NodeMeta& meta = it->second;
    if (mutation.kind == MemoryMutationKind::Accessed) {
        const std::size_t cls = session_class(current_affinity_);
        ++class_counters_[cls].dram_hits;
        ++stratum_counters_[session_stratum(current_affinity_)].dram_hits;
    } else if (mutation.kind == MemoryMutationKind::Removed && tree.contains(mutation.node_id)) {
        const std::size_t cls = session_class(meta.session_id);
        const std::size_t st = session_stratum(meta.session_id);
        if (!tree.node(mutation.node_id).on_storage) {
            ++class_counters_[cls].drops;
            ++stratum_counters_[st].drops;
        } else {
            ++class_counters_[cls].clean_evicts;
            ++stratum_counters_[st].clean_evicts;
        }
    }
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
    auto session_class = [](AffinityId sid) noexcept -> std::size_t {
        const AffinityId base = sid >= 50000000ULL ? (sid - 50000000ULL) : sid;
        return base >= 100000ULL ? 1 : 0;  // 0 = Agent, 1 = Chat
    };
    switch (mutation.kind) {
        case StorageMutationKind::DumpWriteCommitted: {
            const std::size_t t = storage_tier_index(mutation.placement.tier);
            const double waf = effective_waf(mutation.placement.tier, storage_after_commit);
            program_bytes_[t] += mutation.bytes;
            est_nand_bytes_[t] += waf * static_cast<double>(mutation.bytes);
            req_write_blocks_[t] += mutation.nodes.size;
            if (!mutation.nodes.empty()) {
                const auto nit = nodes_.find(mutation.nodes[0]);
                if (nit != nodes_.end() && nit->second.session_id != 0) {
                    auto sit = sessions_.find(nit->second.session_id);
                    if (sit != sessions_.end()) {
                        // Update soft preferred tier to most recent placement (no permanent hard lock).
                        sit->second.preferred_tier = mutation.placement.tier;
                    }
                }
            }
            for (NodeId id : mutation.nodes) {
                auto it = nodes_.find(id);
                if (it != nodes_.end()) {
                    it->second.write_req_seq = current_req_seq_;
                    const std::size_t cls = session_class(it->second.session_id);
                    const std::size_t st = session_stratum(it->second.session_id);
                    if (t == 0) {
                        ++class_counters_[cls].slc_writes;
                        ++stratum_counters_[st].slc_writes;
                    } else {
                        ++class_counters_[cls].tlc_writes;
                        ++stratum_counters_[st].tlc_writes;
                    }
                }
                refresh_node_sets(id, tree);
                refresh_children_root_membership(id, tree);
            }
            break;
        }
        case StorageMutationKind::CapacityTrimCommitted:
        case StorageMutationKind::IdleTrimCommitted: {
            const std::size_t t = storage_tier_index(mutation.source_tier);
            const double alpha = config_.residence_ema_alpha;
            const bool is_idle = (mutation.kind == StorageMutationKind::IdleTrimCommitted);
            for (NodeId id : mutation.nodes) {
                auto it = nodes_.find(id);
                if (it != nodes_.end()) {
                    const std::size_t cls = session_class(it->second.session_id);
                    const std::size_t st = session_stratum(it->second.session_id);
                    if (is_idle) {
                        ++class_counters_[cls].idle_trims;
                        ++stratum_counters_[st].idle_trims;
                    } else {
                        ++class_counters_[cls].cap_trims;
                        ++stratum_counters_[st].cap_trims;
                    }
                }
                if (it != nodes_.end() && it->second.write_req_seq > 0) {
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
        case StorageMutationKind::StorageAccessCommitted: {
            const std::size_t hit_tier = storage_tier_index(mutation.source_tier);
            for (NodeId id : mutation.nodes) {
                auto it = nodes_.find(id);
                if (it != nodes_.end()) {
                    const std::size_t cls = session_class(current_affinity_);
                    const std::size_t st = session_stratum(current_affinity_);
                    if (hit_tier == 0) {
                        ++class_counters_[cls].slc_hits;
                        ++stratum_counters_[st].slc_hits;
                    } else {
                        ++class_counters_[cls].tlc_hits;
                        ++stratum_counters_[st].tlc_hits;
                    }
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

double EverglintJointPolicy::tier_minimax_weight(
    StorageTier tier,
    const StorageView& storage
) const noexcept {
    const double rho_s = region_wear(StorageTier::Slc, storage);
    const double rho_t = region_wear(StorageTier::Tlc, storage);
    const double diff = (tier == StorageTier::Slc) ? (rho_s - rho_t) : (rho_t - rho_s);
    if (!config_.use_smooth_sigmoid) {
        if (diff > 0.0) {
            return 1.0;
        }
        if (diff < 0.0) {
            return 0.0;
        }
        return 0.5;
    }
    const double total = rho_s + rho_t;
    const double z = total > 1e-15 ? std::clamp(6.0 * diff / total, -12.0, 12.0) : 0.0;
    return 1.0 / (1.0 + std::exp(-z));
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
    if (!config_.use_smooth_sigmoid) {
        // Exact hard minimax finite difference (decoupled from wear_eta).
        const double next = tier == StorageTier::Slc
                                ? wear_potential(rho_s + delta, rho_t)
                                : wear_potential(rho_s, rho_t + delta);
        return next - wear_potential(rho_s, rho_t);
    }
    const double minimax_weight = tier_minimax_weight(tier, storage);
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

double EverglintJointPolicy::effective_residence(
    StorageTier tier,
    const StorageView& storage
) const noexcept {
    const std::size_t idx = storage_tier_index(tier);
    const double cap = static_cast<double>(storage.storage().tier(tier).capacity_blocks());
    const double rate = write_blocks_per_req_ema_[idx];
    const double little_res =
        rate > 1e-4 ? std::min(static_cast<double>(config_.horizon_requests) * 2.0, cap / rate)
                    : 0.0;
    if (eviction_samples_[idx] == 0 || residence_ema_[idx] <= 0.0) {
        return little_res;
    }
    if (little_res <= 0.0) {
        return residence_ema_[idx];
    }
    // Anchor observed eviction/TRIM residence EMA with Little's Law capacity turnover scale
    // so proactive TRIM never freezes residence_ema_ at stale cold-start values.
    return std::clamp(residence_ema_[idx], 0.5 * little_res, 2.0 * little_res);
}

double EverglintJointPolicy::survival_prob(
    const NodeMeta& meta,
    StorageTier tier,
    const StorageView& storage
) const noexcept {
    if (!config_.use_survival) {
        return 1.0;
    }
    const double res = effective_residence(tier, storage);
    if (res <= 0.0) {
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
        res,
        config_.prior_reuse_prob,
        config_.prior_weight,
        config_.use_jensen_exact
    );
}

double EverglintJointPolicy::uniform_survival_prob(
    const NodeMeta& meta,
    const StorageView& storage
) const noexcept {
    if (!config_.use_survival) {
        return 1.0;
    }
    const double total_cap = static_cast<double>(
        storage.storage().tier(StorageTier::Slc).capacity_blocks() +
        storage.storage().tier(StorageTier::Tlc).capacity_blocks()
    );
    const double little_global =
        global_write_blocks_per_req_ema_ > 1e-4
            ? std::min(static_cast<double>(config_.horizon_requests) * 2.0,
                       total_cap / global_write_blocks_per_req_ema_)
            : 0.0;
    double res = global_residence_ema_;
    if (global_eviction_samples_ == 0 || res <= 0.0) {
        res = little_global;
    } else if (little_global > 0.0) {
        res = std::clamp(res, 0.5 * little_global, 2.0 * little_global);
    }
    if (res <= 0.0) {
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
        res,
        config_.prior_reuse_prob,
        config_.prior_weight,
        config_.use_jensen_exact
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

std::vector<EverglintJointPolicy::VictimChain>
EverglintJointPolicy::collect_victim_candidates(
    StorageTier tier,
    const std::unordered_set<NodeId>& protected_set,
    const StorageView& storage
) const {
    const RadixTree& tree = storage.tree();
    const auto& leaf_set = ssd_leaf_set_[storage_tier_index(tier)];
    std::vector<VictimChain> candidates;
    candidates.reserve(std::min<std::size_t>(leaf_set.size(), config_.victim_limit));

    std::size_t scanned = 0;
    for (const auto& [recency, leaf_id] : leaf_set) {
        if (scanned >= config_.victim_limit) {
            break;
        }
        ++scanned;
        if (protected_set.count(leaf_id) || !tree.contains(leaf_id)) {
            continue;
        }

        VictimChain chain;
        chain.recency = recency;
        const std::size_t max_len = static_cast<std::size_t>(config_.chunk_blocks);
        std::optional<NodeId> cur = leaf_id;
        double chain_loss = 0.0;
        while (cur.has_value() && chain.nodes_bottom_up.size() < max_len) {
            if (protected_set.count(*cur) || !tree.contains(*cur)) {
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
                                   ? survival_prob(mit->second, tier, storage)
                                   : uniform_survival_prob(mit->second, storage);
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
    return candidates;
}

bool EverglintJointPolicy::select_ssd_victims(
    StorageTier tier,
    std::uint64_t need_blocks,
    const std::vector<VictimChain>& candidates,
    const std::unordered_set<NodeId>& protected_set,
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
    std::unordered_set<NodeId> selected;
    std::uint64_t freed = 0;
    for (const auto& chain : candidates) {
        if (freed >= need_blocks) {
            break;
        }
        // Whole-chain fast path when no overlap with already-selected nodes
        bool clean_chain = true;
        for (NodeId id : chain.nodes_bottom_up) {
            if (selected.count(id)) {
                clean_chain = false;
                break;
            }
        }
        if (clean_chain) {
            for (NodeId id : chain.nodes_bottom_up) {
                selected.insert(id);
                out_victims.push_back(id);
            }
            out_replacement_loss += chain.total_loss;
            freed += chain.nodes_bottom_up.size();
            continue;
        }
        for (NodeId id : chain.nodes_bottom_up) {
            if (!selected.insert(id).second) {
                break;
            }
            const auto& meta = nodes_.at(id);
            const auto [p_e, tau_e] = predict_node(meta);
            static_cast<void>(tau_e);
            const double q_e = (mode_ == Mode::Everglint)
                                   ? survival_prob(meta, tier, storage)
                                   : uniform_survival_prob(meta, storage);
            out_replacement_loss += p_e * q_e * block_recompute_cost(meta);
            out_victims.push_back(id);
            ++freed;
        }
    }

    if (freed < need_blocks) {
        const auto& leaf_set = ssd_leaf_set_[storage_tier_index(tier)];
        std::size_t idx = 0;
        for (const auto& [recency, leaf_id] : leaf_set) {
            ++idx;
            if (idx <= config_.victim_limit) {
                continue;
            }
            if (freed >= need_blocks) {
                break;
            }
            if (protected_set.count(leaf_id) || selected.count(leaf_id) || !tree.contains(leaf_id)) {
                continue;
            }
            std::optional<NodeId> cur = leaf_id;
            while (cur.has_value() && freed < need_blocks) {
                if (protected_set.count(*cur) || selected.count(*cur) || !tree.contains(*cur)) {
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
                                       ? survival_prob(mit->second, tier, storage)
                                       : uniform_survival_prob(mit->second, storage);
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
                static_cast<void>(tau_i);
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
    // Pre-compute sorted SSD victim candidate chains ONCE per tier (disjoint from DRAM-only moving nodes).
    std::array<std::vector<VictimChain>, 2> tier_victim_candidates;
    std::array<bool, 2> tier_candidates_ready{false, false};
    auto get_tier_candidates = [&](StorageTier tier) -> const std::vector<VictimChain>& {
        const std::size_t idx = storage_tier_index(tier);
        if (!tier_candidates_ready[idx]) {
            tier_victim_candidates[idx] = collect_victim_candidates(tier, protected_set, storage);
            tier_candidates_ready[idx] = true;
        }
        return tier_victim_candidates[idx];
    };

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

        const auto parent_opt = tree.parent(root_id);
        const bool parent_on_ssd =
            parent_opt.has_value() && tree.contains(*parent_opt) &&
            tree.node(*parent_opt).on_storage;
        const bool parent_in_slc =
            parent_on_ssd && tree.node(*parent_opt).storage_tier == StorageTier::Slc;

        std::optional<StorageTier> session_tier;
        if (config_.use_session_coherence && root_meta.session_id != 0) {
            const auto sit = sessions_.find(root_meta.session_id);
            if (sit != sessions_.end()) {
                session_tier = sit->second.preferred_tier;
            }
        }

        std::vector<StorageTier> candidate_tiers;
        bool stop_after_first_feasible = false;
        if (mode_ == Mode::Everglint) {
            if (config_.use_session_coherence && parent_in_slc) {
                // Hard prefix-monotonicity: children of an SLC prefix never pin SLC from TLC.
                candidate_tiers = config_.allow_drop
                                      ? std::vector<StorageTier>{StorageTier::Slc}
                                      : std::vector<StorageTier>{StorageTier::Slc, StorageTier::Tlc};
                stop_after_first_feasible = true;
            } else {
                // Soft session hysteresis: ALWAYS evaluate BOTH SLC and TLC so smooth minimax
                // continuously rebalances regional wear across long sessions.
                candidate_tiers = {StorageTier::Slc, StorageTier::Tlc};
            }
        } else if (mode_ == Mode::GandalfTlc) {
            candidate_tiers = {StorageTier::Tlc};
        } else if (mode_ == Mode::GandalfSlc) {
            candidate_tiers = {StorageTier::Slc};
        } else {
            // Gandalf-WearShare: endurance-proportional affinity hash with configurable salt.
            const std::uint64_t key =
                root_meta.session_id == 0 ? root_id : root_meta.session_id;
            constexpr std::uint64_t mask = (1ULL << 53) - 1;
            const double draw =
                static_cast<double>(stable_hash(key, config_.gandalf_hash_salt) & mask) /
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
            if (need > 0 && !select_ssd_victims(
                    tier,
                    need,
                    get_tier_candidates(tier),
                    protected_set,
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
                                        ? survival_prob(meta, tier, storage)
                                        : uniform_survival_prob(meta, storage);
                const double c_i = block_recompute_cost(meta);
                early_evict_loss += p_i * (1.0 - q_ik) * c_i;
            }

            double wear_cost =
                active_wear_lambda_ *
                ((mode_ == Mode::Everglint) ? marginal_wear(tier, write_bytes, storage)
                                            : uniform_marginal_wear(write_bytes, storage));
            if (mode_ == Mode::Everglint && config_.use_session_coherence &&
                session_tier.has_value() && *session_tier == tier) {
                wear_cost *= (1.0 - config_.session_hysteresis);
            }
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
    if (config_.trim_gamma <= 0.0 || active_wear_lambda_ <= 0.0) {
        return result;
    }
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

    constexpr std::size_t kPerTierTrimCap = 512;
    for (StorageTier tier : {StorageTier::Slc, StorageTier::Tlc}) {
        const auto& leaf_set = ssd_leaf_set_[storage_tier_index(tier)];
        if (leaf_set.empty()) {
            continue;
        }
        const double excess_waf = std::max(0.0, effective_waf(tier, storage) - 1.0);
        if (excess_waf <= 0.0) {
            continue;
        }
        const double base_delta =
            (excess_waf * static_cast<double>(block_size)) / region_budget_bytes(tier, storage);
        const double weight =
            (mode_ == Mode::Everglint)
                ? (tier_minimax_weight(tier, storage) + config_.wear_eta)
                : (0.5 + config_.wear_eta);
        const double trim_threshold =
            config_.trim_gamma * active_wear_lambda_ * weight * base_delta;
        if (trim_threshold <= 0.0) {
            continue;
        }

        std::size_t scanned = 0;
        std::size_t tier_trimmed = 0;
        for (const auto& [recency, leaf_id] : leaf_set) {
            if (scanned >= config_.victim_limit || tier_trimmed >= kPerTierTrimCap) {
                break;
            }
            ++scanned;
            if (protected_set.count(leaf_id) || selected.count(leaf_id) || !tree.contains(leaf_id)) {
                continue;
            }
            std::optional<NodeId> cur = leaf_id;
            std::size_t chain_len = 0;
            while (cur.has_value() && chain_len < 512 && tier_trimmed < kPerTierTrimCap) {
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
                                       ? survival_prob(mit->second, tier, storage)
                                       : uniform_survival_prob(mit->second, storage);
                const double val = p_e * q_e * block_recompute_cost(mit->second);
                if (val >= trim_threshold) {
                    break;
                }
                selected.insert(*cur);
                result.emplace_back(tier, *cur);
                ++chain_len;
                ++tier_trimmed;
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
