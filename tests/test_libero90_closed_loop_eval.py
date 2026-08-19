from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/evaluation/eval_libero90_closed_loop.py"
    spec = importlib.util.spec_from_file_location("closed_loop_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_demo_selection_is_limited_to_five_held_out_demos() -> None:
    names = [f"demo_{index}" for index in range(50)]
    assert _module().held_out_demo_name(names, 0) == "demo_45"
    assert _module().held_out_demo_name(names, 4) == "demo_49"
    with pytest.raises(ValueError):
        _module().held_out_demo_name(names, 5)


def test_training_demo_selection_excludes_five_held_out_demos() -> None:
    names = [f"demo_{index}" for index in range(50)]
    assert _module().demo_name(names, "train", 0) == "demo_0"
    assert _module().demo_name(names, "train", 44) == "demo_44"
    with pytest.raises(ValueError):
        _module().demo_name(names, "train", 45)


def test_launcher_uses_two_envs_and_the_300k_checkpoint() -> None:
    script = (ROOT / "jobs/eval_iws_libero90_closed_loop.sbatch").read_text()
    assert "conda activate smolvla" in script
    assert "conda activate /home/youssef.ghallab/miniforge3/envs/iws" in script
    assert "epoch=2-step=300000.ckpt" in script
    assert '--task-indices "${task_indices[@]}"' in script
    assert '--demo-split "$DEMO_SPLIT"' in script
    assert 'PHASE="${PHASE:-all}"' in script


def test_checkpoint_loader_registers_saved_config_resolvers() -> None:
    source = (
        ROOT / "scripts/evaluation/eval_libero90_closed_loop.py"
    ).read_text()
    assert 'OmegaConf.has_resolver("eval")' in source
    assert 'OmegaConf.has_resolver("torch")' in source
