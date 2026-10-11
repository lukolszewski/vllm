# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngramConfig / Qwen4Exp model-state tests, ported from PR #58815.

Covers the `tests/test_config.py` hunk (`host_file_gather` validation) and the
`tests/models/qwen4_exp/test_config.py` hunk (staging in ``prepare_inputs``).
Self-contained: the file-gather fixtures are copied from
``tests/models/qwen4_exp/test_ple.py`` rather than imported from the vLLM test
tree. Everything runs CPU-only.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.common.ngram_embedding as ngram_embedding_module
import vllm.models.qwen4_exp.nvidia.ngram_embedding as nvidia_ngram_embedding_module
from vllm.config import EngramConfig, ModelConfig, ParallelConfig
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEFileGatherEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def _mock_etp_group(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int = 1,
    all_reduce=lambda tensor: tensor,
) -> None:
    """Copy of the helper in tests/models/qwen4_exp/test_ple.py."""
    group = SimpleNamespace(
        rank_in_group=0,
        world_size=world_size,
        all_reduce=all_reduce,
    )
    monkeypatch.setattr(ngram_embedding_module, "get_etp_group", lambda: group)
    monkeypatch.setattr(
        ngram_embedding_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size),
    )


def _patch_ple_construction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engram_config: SimpleNamespace | None = None,
    load_format: str = "auto",
) -> None:
    _mock_etp_group(monkeypatch)
    for module in (embedding_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(load_format=load_format),
        engram_config=engram_config,
    )
    monkeypatch.setattr(
        nvidia_ngram_embedding_module, "get_current_vllm_config", lambda: vllm_config
    )


def _make_file_gather_embedding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fp8: bool = False,
    load_format: str = "auto",
    bind: bool = False,
) -> tuple[Qwen4ExpNGramEmbedding, torch.Tensor]:
    """Build a two-head, 12-token host-file-gather embedding and its saved table.

    The 21-row table is saved to ``tmp_path / "model.safetensors"`` as four
    checkpoint shards of stride 6, so the last shard holds 3 rows.
    """
    _patch_ple_construction(monkeypatch, load_format=load_format)
    method = (
        Qwen4ExpPLEFp8EmbeddingMethod()
        if fp8
        else Qwen4ExpPLEUnquantizedEmbeddingMethod()
    )
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.split_ngram_parts = 4
    module.eos_token_id = 0
    module.ngram_size = 2
    module.heads_per_ngram = 2
    module.register_buffer("layer_multipliers", torch.tensor([3, 5]))
    module.register_buffer("ngram_heads_vocab_sizes", torch.tensor([13, 7]))
    module.register_buffer("ngram_heads_offsets", torch.tensor([0, 13]))
    module.ngram_embedding = Qwen4ExpPLEFileGatherEmbedding(
        21,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        prefix="test.ple_embedding",
        embedding_method=method,
        num_ngram_heads=2,
        max_total_tokens=12,
    )
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    table = torch.arange(42, dtype=torch.float32).reshape(21, 2).to(dtype)
    checkpoint = {
        f"ngram_embedding.shard_{index}.weight": shard.clone()
        for index, shard in enumerate(table.split(6))
    }
    if fp8:
        checkpoint["ngram_embedding.weight_scale"] = torch.tensor([0.25]).bfloat16()
    save_file(checkpoint, tmp_path / "model.safetensors")
    if bind:
        _load_file_shards(module, tmp_path / "model.safetensors")
        module.ngram_embedding.bind_file_shards()
    return module, table


def _load_file_shards(module: Qwen4ExpNGramEmbedding, path: Path) -> set[str]:
    """Load ``path`` through ``load_weights`` as the lazy safetensors iterator does."""
    with safe_open(path, framework="pt") as checkpoint:
        return module.load_weights(
            (name, checkpoint.get_tensor(name))
            for name in checkpoint.keys()  # noqa: SIM118
        )


@pytest.mark.parametrize(
    "option", ["embedding_across_dp", "dp_shared_memory", "host_file_gather"]
)
def test_engram_rejects_elastic_cross_dp(option):
    parallel = ParallelConfig(
        tensor_parallel_size=4,
        data_parallel_size=2,
        distributed_executor_backend="mp",
    )
    parallel.enable_elastic_ep = True
    with pytest.raises(ValueError, match=f"{option}.*elastic EP"):
        EngramConfig(**{option: True}).verify_parallel_config(parallel)


@pytest.mark.parametrize(
    ("architecture", "layer_field", "rocm", "expect_error"),
    [
        pytest.param(
            "DeepseekV41ForCausalLM", "engram_layer_ids", False, True, id="deepseek"
        ),
        pytest.param(
            "Qwen4ExpForCausalLM", "ple_layer_ids", True, True, id="qwen4exp_rocm"
        ),
        pytest.param(
            "Qwen4ExpForCausalLM", "ple_layer_ids", False, False, id="qwen4exp_cuda"
        ),
    ],
)
def test_engram_host_file_gather_requires_cuda_qwen4exp(
    monkeypatch, architecture, layer_field, rocm, expect_error
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: not rocm)
    model = cast(
        ModelConfig,
        SimpleNamespace(
            architecture=architecture,
            hf_text_config=SimpleNamespace(**{layer_field: [1]}),
        ),
    )
    config = EngramConfig(host_file_gather=True)
    if expect_error:
        with pytest.raises(ValueError, match="host_file_gather"):
            config.verify_model_config(model)
    else:
        config.verify_model_config(model)


def test_engram_hash_tracks_execution_options():
    configs = [
        EngramConfig(cpu_offload=True),
        EngramConfig(cpu_offload=False),
        EngramConfig(cpu_offload=True, embedding_across_dp=True),
        EngramConfig(cpu_offload=True, dp_shared_memory=True),
        EngramConfig(host_file_gather=True),
    ]
    assert len({config.compute_hash() for config in configs}) == len(configs)


def test_qwen4_exp_model_state_stages_file_gather_rows(monkeypatch, tmp_path) -> None:
    """Rows are staged from the same PLE inputs the forward receives."""
    module, table = _make_file_gather_embedding(monkeypatch, tmp_path, bind=True)
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_eos_token_id = 0
    model_state.ngram_context = torch.empty((4, 1), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-1, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)
    model_state.file_gather_modules = [module]
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=2,
        idx_mapping=torch.tensor([0, 1]),
        query_start_loc=torch.tensor([0, 4, 6], dtype=torch.int32),
        input_ids=torch.tensor([5, 9, 3, 7, 2, 11]),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([2, 1])),
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[1, 8, 6], [4, 3, 3]])),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    ids = module.compute_ngram_ids(
        input_batch.input_ids,
        model_inputs["query_start_loc"],
        model_inputs["ngram_context"],
    )
    assert torch.equal(module.ngram_embedding._staging[:6], table[ids])


def test_qwen4_exp_model_state_prepares_ngram_context() -> None:
    """Upstream test plus the PR's ``file_gather_modules = []`` line.

    The staging hook reads that attribute unconditionally, so a state with no
    file-gather module must still prepare the same n-gram context.
    """
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)
    model_state.file_gather_modules = []

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
        input_ids=torch.tensor([1, 2, 3]),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc = torch.full((9,), 3, dtype=torch.int32)
    expected_query_start_loc[0] = 0
    expected_query_start_loc[1] = 2
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context = torch.full((8, 3), 99, dtype=torch.int32)
    expected_context[:2] = torch.tensor([[99, 99, 20], [1, 2, 3]])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)

    # Retain the views to detect reallocations as the request layout changes.
    query_start_loc = model_inputs["query_start_loc"]
    ngram_context = model_inputs["ngram_context"]
    input_batch.num_reqs = 1
    input_batch.num_reqs_after_padding = 1
    input_batch.idx_mapping = torch.tensor([0])
    input_batch.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc.fill_(3)
    expected_query_start_loc[0] = 0
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context.fill_(99)
    expected_context[0] = torch.tensor([1, 2, 3])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)
    assert model_inputs["query_start_loc"].data_ptr() == query_start_loc.data_ptr()
    assert model_inputs["ngram_context"].data_ptr() == ngram_context.data_ptr()
