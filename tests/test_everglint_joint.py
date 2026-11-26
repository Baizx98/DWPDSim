import csv

import pytest

from dwpdsim import (
    DWPDSimulator,
    MemoryConfig,
    MemoryPolicyConfig,
    Request,
    SimulationConfig,
    StoragePolicyConfig,
    StorageTierConfig,
)


def make_config(
    *,
    memory_blocks=4,
    slc_blocks=6,
    tlc_blocks=12,
    mode="everglint",
    wear_lambda=1.0e5,
    wear_eta=0.1,
    slc_pe_limit=30000.0,
    tlc_pe_limit=3000.0,
    chunk_blocks=2,
    use_survival=True,
    use_prefix_cost=True,
    allow_drop=True,
    use_value_victim=True,
    admit_storage_hits=False,
):
    block_size = 512
    return SimulationConfig(
        block_size_bytes=block_size,
        memory=MemoryConfig(memory_blocks * block_size),
        slc=StorageTierConfig(slc_blocks * block_size, 1),
        tlc=StorageTierConfig(tlc_blocks * block_size, 1),
        memory_policy=MemoryPolicyConfig(admit_storage_hits=admit_storage_hits),
        storage_policy=StoragePolicyConfig(
            kind="everglint_joint",
            everglint_mode=mode,
            everglint_wear_lambda=wear_lambda,
            everglint_wear_eta=wear_eta,
            everglint_slc_pe_limit=slc_pe_limit,
            everglint_tlc_pe_limit=tlc_pe_limit,
            everglint_chunk_blocks=chunk_blocks,
            everglint_use_survival=use_survival,
            everglint_use_prefix_cost=use_prefix_cost,
            everglint_allow_drop=allow_drop,
            everglint_use_value_victim=use_value_victim,
            logical_fill_fraction=1.0,
        ),
    )


def sample_agent_requests():
    reqs = []
    t = 0
    rid = 1
    # Repeating hot agent sessions + one-shot cold sessions
    for rnd in range(12):
        # Active session 1 (grows from 4 to 8 blocks)
        reqs.append(Request(t, rid, 101, [1001, 1002, 1003, 1004, 1005 + (rnd % 3)]))
        t += 1_000_000_000
        rid += 1
        # Active session 2
        reqs.append(Request(t, rid, 102, [2001, 2002, 2003, 2004, 2005 + (rnd % 2)]))
        t += 1_000_000_000
        rid += 1
        # Ephemeral one-shot session
        base = 10000 + rnd * 10
        reqs.append(Request(t, rid, 500 + rnd, [base, base + 1, base + 2, base + 3]))
        t += 1_000_000_000
        rid += 1
    return reqs


def test_everglint_invariants_capacity_no_migration_and_wear_balance(tmp_path):
    trace_path = tmp_path / "everglint.csv"
    cfg = make_config(wear_lambda=1.0e5)
    with DWPDSimulator(cfg, trace_path) as sim:
        sim.run(sample_agent_requests())
    stats = sim.stats()

    assert stats["memory"]["peak_resident_blocks"] <= 4
    assert stats["storage"]["slc"]["peak_live_bytes"] <= 6 * 512
    assert stats["storage"]["tlc"]["peak_live_bytes"] <= 12 * 512
    assert stats["relocation"]["destination_writes"]["blocks"] == 0
    assert stats["migrations"]["access"]["blocks"] == 0
    assert stats["migrations"]["background"]["blocks"] == 0
    assert stats["migrations"]["capacity"]["blocks"] == 0
    assert stats["storage"]["slc"]["host_write_bytes"] > 0
    assert stats["storage"]["tlc"]["host_write_bytes"] > 0
    assert stats["accesses"]["total_hit_rate"] > 0.3

    with trace_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == stats["trace"]["events"]
    assert all(row["move_id"] == "" for row in rows)


def test_everglint_lambda_controls_write_drop_tradeoff(tmp_path):
    reqs = sample_agent_requests()
    writes_by_lambda = []
    for lam in (0.0, 1.0e7, 1.0e12):
        cfg = make_config(wear_lambda=lam)
        with DWPDSimulator(cfg, tmp_path / f"lam-{lam}.csv") as sim:
            sim.run(reqs)
        st = sim.stats()
        total_writes = (
            st["storage"]["slc"]["host_write_bytes"]
            + st["storage"]["tlc"]["host_write_bytes"]
        )
        writes_by_lambda.append(total_writes)

    assert writes_by_lambda[0] >= writes_by_lambda[1] >= writes_by_lambda[2]
    assert writes_by_lambda[0] > writes_by_lambda[2]


def test_everglint_is_strictly_causal_no_future_leakage(tmp_path):
    reqs = sample_agent_requests()
    prefix = reqs[:20]
    extended = prefix + [Request(999_000_000_000, 9999, 999, [99901, 99902, 99903])]

    p1 = tmp_path / "prefix.csv"
    p2 = tmp_path / "extended.csv"
    with DWPDSimulator(make_config(), p1) as s1:
        s1.run(prefix)
    with DWPDSimulator(make_config(), p2) as s2:
        s2.run(extended)

    rows1 = list(csv.DictReader(p1.open(encoding="utf-8")))
    rows2 = list(csv.DictReader(p2.open(encoding="utf-8")))
    assert rows2[: len(rows1)] == rows1


@pytest.mark.parametrize("mode", ["everglint", "gandalf_tlc", "gandalf_slc", "gandalf_wear_share"])
def test_all_joint_modes_and_ablations_run_cleanly(tmp_path, mode):
    reqs = sample_agent_requests()
    cfg = make_config(mode=mode, wear_lambda=1.0e4)
    with DWPDSimulator(cfg, tmp_path / f"{mode}.csv") as sim:
        sim.run(reqs)
    st = sim.stats()
    assert st["memory"]["peak_resident_blocks"] <= 4
    assert st["relocation"]["destination_writes"]["blocks"] == 0
    if mode == "gandalf_tlc":
        assert st["storage"]["slc"]["host_write_bytes"] == 0
        assert st["storage"]["tlc"]["host_write_bytes"] > 0
    elif mode == "gandalf_slc":
        assert st["storage"]["tlc"]["host_write_bytes"] == 0
        assert st["storage"]["slc"]["host_write_bytes"] > 0


def test_everglint_ablations_run_cleanly(tmp_path):
    reqs = sample_agent_requests()
    for name, kwargs in [
        ("no_survival", {"use_survival": False}),
        ("no_eta", {"wear_eta": 0.0}),
        ("no_prefix_cost", {"use_prefix_cost": False}),
        ("no_drop", {"allow_drop": False}),
        ("lru_victim", {"use_value_victim": False}),
    ]:
        cfg = make_config(**kwargs)
        with DWPDSimulator(cfg, tmp_path / f"{name}.csv") as sim:
            sim.run(reqs)
        st = sim.stats()
        assert st["memory"]["peak_resident_blocks"] <= 4
        if name == "no_drop":
            assert st["memory"]["drop_blocks"] == 0


def test_everglint_new_ablations_promotion_dual_and_measurement_window(tmp_path):
    reqs = sample_agent_requests()
    for name, extra in [
        ("single_stratum", {"everglint_use_cadence_strata": False}),
        ("plugin_mean", {"everglint_use_jensen_exact": False}),
        ("no_coherence", {"everglint_use_session_coherence": False}),
        ("hard_sigmoid", {"everglint_use_smooth_sigmoid": False}),
        ("promote_hot", {"everglint_promote_hit_threshold": 2}),
        ("closed_loop", {"everglint_target_life_years": 1.0, "everglint_dual_step_kappa": 0.25, "everglint_dual_window_ns": 5_000_000_000}),
    ]:
        cfg = SimulationConfig(
            block_size_bytes=512,
            memory=MemoryConfig(4 * 512),
            slc=StorageTierConfig(6 * 512, 1),
            tlc=StorageTierConfig(12 * 512, 1),
            measurement_start_ns=10_000_000_000,
            memory_policy=MemoryPolicyConfig(profile=True),
            storage_policy=StoragePolicyConfig(
                kind="everglint_joint",
                everglint_mode="everglint",
                everglint_wear_lambda=1.0e5,
                everglint_chunk_blocks=2,
                logical_fill_fraction=1.0,
                **extra,
            ),
        )
        with DWPDSimulator(cfg, tmp_path / f"{name}.csv") as sim:
            sim.run(reqs)
            perf = sim._core.memory_performance()
            ext = sim._core.extended_access_stats()
            diag = sim._core.everglint_diagnostics()
        st = sim.stats()
        assert st["memory"]["peak_resident_blocks"] <= 4
        assert ext["compute_cost_triangular"] <= ext["compute_cost"]
        assert ext["compute_cost_triangular"] > 0
        mw = ext["measurement_window"]
        assert 0 < mw["requests"] < st["accesses"]["requests"]
        assert perf["decisions"] > 0 and perf["decision_ns"] > 0
        assert "class_counters" in diag and "dual_trace" in diag
        if name == "closed_loop":
            assert len(diag["dual_trace"]) > 0
