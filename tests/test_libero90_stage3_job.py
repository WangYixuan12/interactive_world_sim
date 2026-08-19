import os
from pathlib import Path
import subprocess

from hydra.core.override_parser.overrides_parser import OverridesParser


def test_stage3_job_is_resumable_and_loads_stage2(tmp_path: Path) -> None:
    path = Path("jobs/train_iws_libero90_stage3.sbatch")
    script = path.read_text()

    for setting in (
        "#SBATCH --time=12:00:00",
        "#SBATCH --gres=gpu:1",
        "dataset.horizon=1 dataset.val_horizon=200",
        "experiment.training.batch_size=$TRAIN_BATCH_SIZE",
        "experiment.validation.limit_batch=1",
        "algorithm.val_render=false",
        "algorithm.training_stage=3",
        'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"',
        '"algorithm.load_ae=\'$STAGE2_CKPT\'"',
        "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
    ):
        assert setting in script

    checkpoint = "/tmp/epoch=2-step=300000.ckpt"
    override = OverridesParser.create().parse_overrides(
        [f"algorithm.load_ae='{checkpoint}'"]
    )[0]
    assert override.value() == checkpoint

    result = subprocess.run(
        ["bash", path],
        env={**os.environ, "STAGE2_CKPT": str(tmp_path / "missing.ckpt")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "STAGE2_CKPT must be a regular file" in result.stderr
