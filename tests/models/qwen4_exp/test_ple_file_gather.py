# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-file-gather PLE tests, ported from PR #58815 (`test_ple.py` hunk).

Self-contained: the helpers the PR's tests rely on (`_mock_etp_group` and the
file-gather fixtures) are copied from `tests/models/qwen4_exp/test_ple.py`, so
this file imports nothing from the vLLM test tree. Everything runs CPU-only --
the rows are read on the host with `preadv` and staged into a device buffer, so
no test needs a CUDA tensor. `torch.empty(..., pin_memory=...)` is only reached
with `pin_memory=False` here because the weight device is CPU.
"""

import mmap
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load, load_file, save_file
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.common.ngram_embedding as ngram_embedding_module
import vllm.models.qwen4_exp.nvidia.ngram_embedding as nvidia_ngram_embedding_module
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEFileGatherEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)


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


def _dev_zero_shards() -> list[tuple[str, torch.Tensor]]:
    """Return shards in a ``/dev/zero`` mapping, which is how pinned memory maps."""
    with open("/dev/zero", "r+b") as dev_zero:
        mapping = mmap.mmap(dev_zero.fileno(), mmap.PAGESIZE)
    table = torch.frombuffer(mapping, dtype=torch.uint8)[:84].view(torch.bfloat16)
    return [
        (f"ngram_embedding.shard_{index}.weight", shard)
        for index, shard in enumerate(table.reshape(21, 2).split(6))
    ]


@pytest.mark.parametrize(
    ("engram_config", "backend", "rows"),
    [
        (
            SimpleNamespace(host_file_gather=True, cpu_offload=True),
            Qwen4ExpPLEFileGatherEmbedding,
            0,
        ),
        (
            SimpleNamespace(host_file_gather=False, cpu_offload=False),
            Qwen4ExpPLEDeviceEmbedding,
            24,
        ),
        (None, Qwen4ExpPLEDeviceEmbedding, 24),
    ],
)
def test_ngram_embedding_selects_backend_from_engram_config(
    monkeypatch, engram_config, backend, rows
) -> None:
    """File gather wins with no rows; ``cpu_offload=False`` keeps rows on device."""
    _patch_ple_construction(monkeypatch, engram_config=engram_config)
    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=2,
        eos_token_id=0,
        vocab_size=64,
        ngram_vocab_size_base=8,
        make_ngram_vocab_size_divisible_by=1,
    )

    module = Qwen4ExpNGramEmbedding(config, 4, 0, 4, data_parallel_rank=0, prefix="t")

    assert tuple(module.ngram_embedding.weight.shape) == (rows, 2)
    assert isinstance(module.ngram_embedding, backend)


def test_ngram_embedding_default_cpu_offload_selects_pinned_backend(
    monkeypatch,
) -> None:
    """Default ``cpu_offload=True`` selects pinned storage and requires UVA."""
    _patch_ple_construction(
        monkeypatch,
        engram_config=SimpleNamespace(host_file_gather=False, cpu_offload=True),
    )
    monkeypatch.setattr(ngram_embedding_module, "is_uva_available", lambda: False)
    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=2,
        eos_token_id=0,
        vocab_size=64,
        ngram_vocab_size_base=8,
        make_ngram_vocab_size_divisible_by=1,
    )

    with pytest.raises(RuntimeError, match="CPU offload requires UVA"):
        Qwen4ExpNGramEmbedding(config, 4, 0, 4, data_parallel_rank=0, prefix="t")


@pytest.mark.parametrize(
    ("read_shards", "error"),
    [
        pytest.param(
            lambda path: [("ngram_embedding.shard_0.weight", torch.zeros(6, 2))],
            "dtype torch.float32 must match the embedding dtype torch.bfloat16",
            id="wrong_dtype",
        ),
        pytest.param(
            lambda path: [*load_file(path).items()] * 2,
            "Duplicate PLE embedding shard 0",
            id="duplicate_shard",
        ),
        pytest.param(
            lambda path: [s for s in load_file(path).items() if "_2." not in s[0]],
            "PLE shards cover 15 of 21 rows",
            id="missing_shard",
        ),
        pytest.param(lambda path: [], "PLE shards cover 0 of 21 rows", id="no_shards"),
        pytest.param(
            lambda path: [(n, t.clone()) for n, t in load_file(path).items()],
            "not a file-backed view",
            id="anonymous",
        ),
        pytest.param(
            lambda path: load(path.read_bytes()).items(),
            "not a file-backed view",
            id="heap",
        ),
        pytest.param(
            lambda path: _dev_zero_shards(), "not a file-backed view", id="dev_zero"
        ),
    ],
)
def test_file_gather_rejects_shards_that_cannot_serve_every_row_from_a_file(
    monkeypatch, tmp_path, read_shards, error
) -> None:
    """A real load must fail instead of serving zeros or a copied table."""
    module, _ = _make_file_gather_embedding(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match=error):
        module.load_weights(read_shards(tmp_path / "model.safetensors"))
        module.ngram_embedding.bind_file_shards()


@pytest.mark.parametrize("fp8", [False, True])
def test_file_gather_bind_releases_the_loader_views(monkeypatch, tmp_path, fp8) -> None:
    """Fails if the shard views (and their file mapping) survive bind."""
    module, _ = _make_file_gather_embedding(monkeypatch, tmp_path, fp8=fp8)
    embedding = module.ngram_embedding

    loaded = _load_file_shards(module, tmp_path / "model.safetensors")
    embedding.quant_method.process_weights_after_loading(embedding)
    embedding.quant_method.process_weights_after_loading(embedding)

    assert loaded == {"ngram_embedding.weight"} | (
        {"ngram_embedding.weight_scale"} if fp8 else set()
    )
    assert embedding._bound
    real = os.path.realpath(tmp_path / "model.safetensors")
    assert real not in Path("/proc/self/maps").read_text()


def test_file_gather_binds_shards_opened_through_a_symlink(
    monkeypatch, tmp_path
) -> None:
    """HF cache snapshots link to suffix-less blobs, which is the mapped path."""
    module, _ = _make_file_gather_embedding(monkeypatch, tmp_path)
    blob = (tmp_path / "model.safetensors").rename(tmp_path / "blob")
    snapshot = tmp_path / "snapshot.safetensors"
    snapshot.symlink_to(blob)
    _load_file_shards(module, snapshot)

    module.ngram_embedding.bind_file_shards()

    assert module.ngram_embedding._bound


def test_file_gather_rejects_reload_and_stays_poisoned(monkeypatch, tmp_path) -> None:
    module, table = _make_file_gather_embedding(monkeypatch, tmp_path, bind=True)

    with pytest.raises(RuntimeError, match="does not support weight reload"):
        module.load_weights([("ngram_embedding.shard_0.weight", table[:6])])
    with pytest.raises(RuntimeError, match="does not support weight reload"):
        module.ngram_embedding.bind_file_shards()
    with pytest.raises(RuntimeError, match="unbound or was reloaded"):
        module.ngram_embedding.stage_rows(1)


@pytest.mark.parametrize("bad_id", [21, 24, -1])
def test_file_gather_out_of_range_id_keeps_previous_staged_rows(
    monkeypatch, tmp_path, bad_id
) -> None:
    """Out-of-range ids fail before the H2D copy, so no wrong row is served."""
    module, table = _make_file_gather_embedding(monkeypatch, tmp_path, bind=True)
    embedding = module.ngram_embedding
    ids = torch.tensor(
        [
            [20, 0],
            [19, 1],
            [18, 2],
            [17, 3],
            [16, 4],
            [15, 5],
            [14, 6],
            [13, 7],
            [12, 8],
            [11, 9],
            [10, 10],
        ]
    )
    embedding._host_ids[:11] = ids
    embedding.stage_rows(11)
    embedding._host_ids[5, 1] = bad_id

    with pytest.raises(IndexError):
        embedding.stage_rows(11)

    assert torch.equal(embedding._staging[:11], table[ids])


def test_file_gather_short_read_raises_and_keeps_previous_staged_rows(
    monkeypatch, tmp_path
) -> None:
    """A truncated shard file raises before any row reaches the staging buffer."""
    module, table = _make_file_gather_embedding(monkeypatch, tmp_path, bind=True)
    embedding = module.ngram_embedding
    ids = torch.arange(1, 21).reshape(10, 2)
    embedding._host_ids[:10] = ids
    embedding.stage_rows(10)
    path = tmp_path / "model.safetensors"
    os.truncate(path, path.stat().st_size - 4)

    with pytest.raises(ValueError, match="truncated at offset"):
        embedding.stage_rows(10)

    assert torch.equal(embedding._staging[:10], table[ids])


def test_file_gather_reads_ahead_the_coalesced_pages_of_staged_rows(
    monkeypatch, tmp_path
) -> None:
    """Rows start 34 bytes before page 2 of a file mapped from page 1.

    Rows 0-7 are in page 1, row 8 straddles pages 1 and 2, and the rows of shard 1
    arrive unsorted.
    """
    module, table = _make_file_gather_embedding(monkeypatch, tmp_path)
    embedding, page = module.ngram_embedding, mmap.PAGESIZE
    start = 2 * page - 34
    path = tmp_path / "table.bin"
    path.write_bytes(bytes(start) + table.view(torch.uint8).numpy().tobytes())
    with open(path, "rb") as file:
        mapping = mmap.mmap(file.fileno(), 0, offset=page, access=mmap.ACCESS_COPY)
    rows = torch.frombuffer(mapping, dtype=torch.uint8)[start - page :]
    module.load_weights(
        (f"ngram_embedding.shard_{index}.weight", shard)
        for index, shard in enumerate(rows.view(table.dtype).reshape(21, 2).split(6))
    )
    embedding.bind_file_shards()
    calls = []
    monkeypatch.setattr(
        nvidia_ngram_embedding_module.os,
        "posix_fadvise",
        lambda fd, *args: calls.append((os.fstat(fd).st_ino, *args)),
    )
    ids = torch.tensor([[0, 1], [8, 7]])
    embedding._host_ids[:2] = ids

    embedding.stage_rows(2)

    inode, willneed = path.stat().st_ino, os.POSIX_FADV_WILLNEED
    assert calls == [(inode, page, page, willneed), (inode, page, 2 * page, willneed)]
    assert torch.equal(embedding._staging[:2], table[ids])


def test_file_gather_dummy_load_stages_zeros(monkeypatch, tmp_path) -> None:
    module, _ = _make_file_gather_embedding(monkeypatch, tmp_path, load_format="dummy")
    embedding = module.ngram_embedding
    embedding.quant_method.process_weights_after_loading(embedding)
    embedding._host_rows.view(torch.uint8).fill_(0xFF)
    embedding._staging.view(torch.uint8).fill_(0xFF)

    embedding.stage_rows(4)

    staged = embedding._staging.view(torch.uint8)
    assert torch.all(staged[:4] == 0)
    assert torch.all(staged[4:] == 0xFF)


def test_file_gather_forward_reads_staging_without_host_work(
    monkeypatch, tmp_path
) -> None:
    """The forward must be a fixed-address read so FULL CUDA graphs replay it."""
    module, _ = _make_file_gather_embedding(monkeypatch, tmp_path)
    staging = module.ngram_embedding._staging
    staging.view(torch.uint8).fill_(0xFF)
    step = (torch.arange(6), torch.tensor([0, 6]), torch.tensor([[1]]))
    hidden_states = torch.zeros(6, 8, dtype=torch.bfloat16)

    assert module.start_prefetch(hidden_states, *step) is None
    output = module(hidden_states, *step)

    assert torch.all(staging.view(torch.uint8) == 0xFF)
    assert output.data_ptr() == staging.data_ptr()
    assert output.shape == (6, 4)


def _file_gather_modules(model: nn.Module) -> list[Qwen4ExpNGramEmbedding]:
    """Copy of the comprehension in vllm/models/qwen4_exp/nvidia/model_state.py.

    ``model_state`` needs a ``VllmConfig`` to import, so its logic is duplicated
    here instead of imported::

        self.file_gather_modules = [
            module
            for module in model.modules()
            if isinstance(module, Qwen4ExpNGramEmbedding)
            and isinstance(module.ngram_embedding, Qwen4ExpPLEFileGatherEmbedding)
        ]
    """
    return [
        module
        for module in model.modules()
        if isinstance(module, Qwen4ExpNGramEmbedding)
        and isinstance(module.ngram_embedding, Qwen4ExpPLEFileGatherEmbedding)
    ]


def test_pp_rank_without_ple_has_no_file_gather_modules(
    monkeypatch, tmp_path
) -> None:
    """A pipeline rank that holds no PLE module must stage nothing.

    Under PP the PLE layer lives on one stage only; every other rank builds the
    same model minus the embedding, so the ``model_state`` comprehension -- the
    only thing that decides whether ``prepare_inputs`` pays for a host staging
    pass -- has to come back empty.
    """
    # A rank with decoder layers but no PLE embedding.
    without_ple = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    assert _file_gather_modules(without_ple) == []

    # A rank with a PLE module whose rows live on the device, not in files.
    device_module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(device_module)
    _patch_ple_construction(monkeypatch)
    device_module.ngram_embedding = Qwen4ExpPLEDeviceEmbedding(
        21,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        prefix="test.ple_embedding",
        embedding_method=Qwen4ExpPLEUnquantizedEmbeddingMethod(),
        num_ngram_heads=2,
        max_total_tokens=12,
    )
    assert _file_gather_modules(nn.Sequential(device_module)) == []

    # A bare nn.Module that merely carries a file-gather table is not a PLE
    # module, so the first isinstance half of the comprehension still filters it.
    lookalike = nn.Module()
    lookalike.ngram_embedding = _make_file_gather_embedding(
        monkeypatch, tmp_path, bind=True
    )[0].ngram_embedding
    assert _file_gather_modules(lookalike) == []

    # Positive control: the comprehension is not vacuously empty.
    gathered = _make_file_gather_embedding(monkeypatch, tmp_path, bind=True)[0]
    assert _file_gather_modules(nn.Sequential(gathered)) == [gathered]
