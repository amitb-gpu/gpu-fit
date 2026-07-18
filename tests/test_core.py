"""Sanity checks on the fit / throughput / cost model."""

from gpu_fit import core


def test_data_loads():
    assert core.load_gpus()
    assert core.load_models()


def test_weight_size_8b_fp16():
    m = core.find_model("llama-3.1-8b")
    assert m is not None
    # 8.03B params * 2 bytes ~= 15 GiB
    assert 14.5 < core.weight_gib(m, "fp16") < 15.5
    # int4 is a quarter of fp16
    assert abs(core.weight_gib(m, "int4") * 4 - core.weight_gib(m, "fp16")) < 1e-6


def test_kv_cache_scales_linearly():
    m = core.find_model("llama-3.1-8b")
    kv1 = core.kv_cache_gib(m, ctx=8192, batch=1)
    kv2 = core.kv_cache_gib(m, ctx=8192, batch=2)
    kv_long = core.kv_cache_gib(m, ctx=16384, batch=1)
    assert abs(kv2 - 2 * kv1) < 1e-6
    assert abs(kv_long - 2 * kv1) < 1e-6


def test_8b_fits_on_4090_fp16():
    m = core.find_model("llama-3.1-8b")
    g = core.find_gpu("4090")
    f = core.best_fit_for_gpu(g, m, ctx=8192, batch=1)
    assert f is not None
    assert f.quant == "fp16"
    assert f.tp == 1
    assert f.tok_s > 0


def test_70b_needs_quant_or_tp_on_single_80gb():
    m = core.find_model("llama-3.1-70b")
    g = core.find_gpu("a100 80gb")
    # 70B fp16 ~= 131 GiB weights, cannot fit fp16 on one 80GB card
    fp16 = core.best_fit_for_gpu(g, m, ctx=8192, batch=1, quant="fp16", allow_tp=False)
    assert fp16 is None
    # but auto (int8/int4) or TP should find something
    auto = core.best_fit_for_gpu(g, m, ctx=8192, batch=1, quant="auto")
    assert auto is not None


def test_recommend_ranks_and_filters_budget():
    m = core.find_model("llama-3.1-8b")
    cheap = core.recommend(m, budget_per_hr=0.50, rank="cost")
    assert cheap
    # every returned priced GPU respects the budget
    for f in cheap:
        if f.usd_per_hr is not None:
            assert f.usd_per_hr <= 0.50
    # cost ranking: first priced entry is cheapest $/Mtok
    priced = [f.cost_mtok for f in cheap if f.cost_mtok is not None]
    assert priced == sorted(priced)


def test_cost_per_mtok_positive():
    m = core.find_model("mistral-7b")
    g = core.find_gpu("a10g")
    f = core.best_fit_for_gpu(g, m, ctx=4096, batch=1)
    assert f.cost_mtok is not None and f.cost_mtok > 0


def test_vllm_command_includes_tp_when_sharded():
    m = core.find_model("llama-3.1-70b")
    g = core.find_gpu("a100 40gb")
    f = core.best_fit_for_gpu(g, m, ctx=8192, batch=1, quant="auto")
    cmd = core.vllm_command(m, f, 8192)
    assert "vllm serve" in cmd
    if f.tp > 1:
        assert "tensor-parallel-size" in cmd
