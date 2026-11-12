"""Automated Tests for Phase VI Part 1: Distributed, Memory & Optimiser Foundations (Steps 47–50)."""

from __future__ import annotations

import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rootism.config import (
    RootismDistributedConfig,
    RootismMemoryConfig,
    RootismModelConfig,
    RootismOptimizerConfig,
    RootismTrainingConfig,
)
from rootism.core.token_accounting import TokenAccountingLedger
from rootism.model.transformer import RootismForCausalLM
from rootism.training import (
    PrecisionAndMemoryController,
    TrainingStepExecutor,
    build_optimizer,
    build_tensor_parallel_plan,
    detect_hardware_capabilities,
    estimate_training_infrastructure,
    initialize_distributed_context,
    partition_layers_for_pipeline,
    wrap_model_for_distributed,
)


def test_hardware_detection_and_infrastructure_estimates_across_gpu_tiers() -> None:
    caps = detect_hardware_capabilities()
    assert caps.cpu_logical_cores >= 1
    assert caps.host_ram_gib > 0

    est_h100 = estimate_training_infrastructure(gpu_preset="h100_80gb_sxm", num_gpus=64)
    assert est_h100.fits_in_gpu_vram is True
    assert est_h100.target_training_tokens == 650_000_000_000
    assert est_h100.kv_cache_32k_gib == 2.5
    assert est_h100.estimated_training_days > 0

    est_4090 = estimate_training_infrastructure(
        gpu_preset="rtx_4090_24gb",
        num_gpus=16,
        sequence_length=2048,
        micro_batch_size=1,
        activation_checkpointing="full",
    )
    assert est_4090.recommended_strategy == "fsdp_zero3"
    assert est_4090.fits_in_gpu_vram is True

    # Tensor & pipeline parallel planners on canonical 40-layer Rootism-3B
    prod_cfg = RootismModelConfig()
    tp_plan = build_tensor_parallel_plan(prod_cfg, tp_size=4)
    assert len(tp_plan) == 40 * 7
    pp_stages = partition_layers_for_pipeline(prod_cfg, pp_size=8)
    assert len(pp_stages) == 8
    assert all(len(s["layer_indices"]) == 5 for s in pp_stages)


def test_selective_and_full_activation_checkpointing_preserve_exact_gradients() -> None:
    cfg = RootismModelConfig.tiny_verification(
        vocab_size=512, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16, intermediate_size=128
    )
    torch.manual_seed(777)
    m_base = RootismForCausalLM(cfg)
    m_sel = RootismForCausalLM(cfg)
    m_full = RootismForCausalLM(cfg)
    m_sel.load_state_dict(m_base.state_dict())
    m_full.load_state_dict(m_base.state_dict())

    m_sel.configure_activation_checkpointing("selective", every_n_layers=2)
    m_full.configure_activation_checkpointing("full")

    ids = torch.randint(4, 512, (2, 24))
    for m in (m_base, m_sel, m_full):
        out = m(input_ids=ids, labels=ids, return_logits=False)
        assert out.loss is not None
        out.loss.backward()

    for (_, p_b), (_, p_s), (_, p_f) in zip(
        m_base.named_parameters(), m_sel.named_parameters(), m_full.named_parameters(), strict=True
    ):
        assert p_b.grad is not None and p_s.grad is not None and p_f.grad is not None
        assert torch.allclose(p_b.grad, p_s.grad, atol=1e-6)
        assert torch.allclose(p_b.grad, p_f.grad, atol=1e-6)


def test_optimizer_parameter_grouping_and_fp32_master_adamw_on_bf16_weights() -> None:
    cfg = RootismModelConfig.tiny_verification(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, intermediate_size=128)
    model = RootismForCausalLM(cfg).to(torch.bfloat16)

    opt_cfg = RootismOptimizerConfig(
        optimizer_type="rootism_fp32_master_adamw",
        learning_rate=1e-3,
        weight_decay=0.1,
    )
    opt, summary = build_optimizer(model, opt_cfg)
    assert summary.optimizer_class == "RootismFP32MasterAdamW"
    assert summary.non_decayed_parameter_count > 0
    assert (
        summary.decayed_parameter_count + summary.non_decayed_parameter_count
        == cfg.analytical_parameter_count()
    )

    # Verify genuine loss reduction on repeated pattern learning with BF16 weights + FP32 master optimizer
    pattern = torch.tensor([[1, 10, 20, 30, 40, 10, 20, 30, 40, 2]], dtype=torch.long)
    losses: list[float] = []
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=pattern, labels=pattern, return_logits=False)
        assert out.loss is not None
        losses.append(float(out.loss.item()))
        out.loss.backward()
        opt.step()

    assert losses[-1] < losses[0] - 1.0


def test_training_step_executor_gradient_accumulation_and_token_accounting() -> None:
    model_cfg = RootismModelConfig.tiny_verification(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, intermediate_size=128)
    train_cfg = RootismTrainingConfig(
        verification_mode=True,
        target_training_tokens=10_000,
        sequence_length=64,
        micro_batch_size=2,
        memory=RootismMemoryConfig(precision="fp32", activation_checkpointing="selective", gradient_accumulation_steps=2),
    )
    model = RootismForCausalLM(model_cfg)
    mem_ctrl = PrecisionAndMemoryController(train_cfg.memory)
    mem_ctrl.apply_model_memory_optimizations(model)
    dist_ctx = initialize_distributed_context(train_cfg.distributed)
    opt, _ = build_optimizer(model, train_cfg.optimizer)
    ledger = TokenAccountingLedger(target_training_tokens=10_000, verification_mode=True)

    executor = TrainingStepExecutor(model, opt, mem_ctrl, dist_ctx, ledger, train_cfg)

    seq = torch.tensor([[1, 12, 24, 36, 48, 60, 72, 2, 0, 0]], dtype=torch.long)
    mbs = [
        {"input_ids": seq.repeat(2, 1), "is_repeated_epoch": False},
        {"input_ids": seq.repeat(2, 1), "is_repeated_epoch": True},
    ]

    r1 = executor.execute_global_step(mbs)
    r2 = executor.execute_global_step(mbs)
    r3 = executor.execute_global_step(mbs)

    assert r1.step_valid and r2.step_valid and r3.step_valid
    assert r3.mean_loss < r1.mean_loss
    # Each sequence has length 10 -> 9 shift targets, 2 trailing pad tokens (0) -> 7 active tokens per row
    # 2 rows * 2 microbatches = 28 active tokens per step * 3 steps = 84 processed_training_tokens!
    assert r3.token_snapshot.processed_training_tokens == 84
    assert r3.token_snapshot.padding_and_masked_tokens == (36 - 28) * 3
    assert r3.token_snapshot.repeated_epoch_tokens == 14 * 3


def _distributed_worker(rank: int, world_size: int, port: int, strategy: str, return_dict: dict) -> None:
    dist_cfg = RootismDistributedConfig(strategy=strategy, backend="gloo")  # type: ignore[arg-type]
    ctx = initialize_distributed_context(
        dist_cfg,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        local_rank=rank,
    )
    try:
        torch.manual_seed(100)
        cfg = RootismModelConfig.tiny_verification(
            vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, intermediate_size=128
        )
        raw_model = RootismForCausalLM(cfg)
        wrapped = wrap_model_for_distributed(raw_model, ctx, dist_cfg)
        train_cfg = RootismTrainingConfig(
            verification_mode=True,
            target_training_tokens=5_000,
            memory=RootismMemoryConfig(precision="fp32", activation_checkpointing="none"),
            distributed=dist_cfg,
        )
        mem = PrecisionAndMemoryController(train_cfg.memory)
        opt, _ = build_optimizer(wrapped, train_cfg.optimizer)
        ledger = TokenAccountingLedger(target_training_tokens=5_000, verification_mode=True)
        executor = TrainingStepExecutor(wrapped, opt, mem, ctx, ledger, train_cfg)

        # Rank-specific batch to verify cross-rank gradient & token count synchronization
        batch = torch.full((2, 9), fill_value=10 + rank * 5, dtype=torch.long)
        res = executor.execute_global_step([{"input_ids": batch, "is_repeated_epoch": False}])
        return_dict[rank] = {
            "valid": res.step_valid,
            "global_processed_tokens": res.token_snapshot.processed_training_tokens,
            "loss": res.mean_loss,
        }
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_real_two_worker_distributed_ddp_and_fsdp_gloo_execution() -> None:
    ctx_mp = mp.get_context("spawn")
    for strat in ("ddp", "fsdp_zero2", "fsdp_zero3"):
        port = _find_free_port()
        manager = ctx_mp.Manager()
        ret = manager.dict()
        mp.spawn(
            _distributed_worker,
            args=(2, port, strat, ret),
            nprocs=2,
            join=True,
        )
        assert ret[0]["valid"] is True and ret[1]["valid"] is True
        # 2 ranks * 2 batch * (9 - 1) targets = 32 global tokens synchronized across both ranks!
        assert ret[0]["global_processed_tokens"] == 32
        assert ret[1]["global_processed_tokens"] == 32
