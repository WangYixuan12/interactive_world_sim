# Task 3 Report: A100 attention fallback

## Status

Implemented the Stage 2 smoke blocker fix. A100 attention now tries Flash first
and permits PyTorch's Math backend as a fallback. The existing large-batch Math
override and non-A100 Math/Efficient backend selection are unchanged.

No Slurm job was submitted.

## Root cause

Smoke job `153131` loaded the Stage 1 checkpoint, initialized the CUDA-enabled
Stage 2 trainer on an A100, and reached the first optimizer step. Its float32
query, key, and value tensors were not eligible for Flash attention. Because
`Attention.__init__` configured A100s with only
`SDPBackend.FLASH_ATTENTION`, PyTorch had no eligible fallback and raised:

```text
Expected query, key and value to all be of dtype: {Half, BFloat16}.
RuntimeError: No available kernel. Aborting execution.
```

All temporal and non-linear spatial attention blocks route through the shared
`Attention` class, so the root fix belongs in its A100 backend list.

## Change

- Added `SDPBackend.MATH` after `SDPBackend.FLASH_ATTENTION` in the A100 CUDA
  backend list.
- Added `tests/test_attention.py`, which monkeypatches
  `torch.cuda.get_device_properties` to A100 compute capability 8.0,
  constructs the real `Attention`, and asserts the ordered backend list.

The mutation protected by the test is removing or reordering the Math fallback
in the A100 branch.

## TDD evidence

The focused test was added before the production change and run with:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest \
  tests/test_attention.py::test_a100_attention_falls_back_to_math -q
```

Red result: `1 failed in 1.42s`. The assertion showed actual
`[SDPBackend.FLASH_ATTENTION]` versus expected
`[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]`.

After the one-line production change, the same command passed:

```text
1 passed in 1.50s
```

## Verification

The existing Stage 2 focused suites passed:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest \
  tests/test_libero90_dataset.py tests/test_libero90_stage1_job.py \
  tests/test_libero90_stage2_job.py -q
```

Result: `10 passed in 6.21s`.

The final combined regression run included the new attention test and all
three suites above:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest \
  tests/test_attention.py tests/test_libero90_dataset.py \
  tests/test_libero90_stage1_job.py tests/test_libero90_stage2_job.py -q
```

Final result: `11 passed`.

`git diff --check` also exited successfully with no output before the final
verification pass.

## Concerns and next gate

The unit regression proves A100 backend ordering without requiring CUDA and
the existing suites prove the dataset and launchers did not regress. A fresh
A100 smoke job is still required to prove PyTorch selects Math for the actual
float32 Stage 2 tensors and that training plus validation complete. That job
was intentionally not submitted in this task.
