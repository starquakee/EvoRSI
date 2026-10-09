# Prepare task inputs for Evo, SFT rollout and RL

First obtain complete packages using the [dataset instructions](https://huggingface.co/datasets/FrontisAI/OpenMLE-Tasks).
Download `built_task` archives directly, or obtain the upstream data and build a
`recipe`. Both routes must retain `info/`, `data/` and `utils/`.

## Prepare one task

Run from `OpenRSI/OpenMLE-Gym`:

```bash
uv sync --no-editable --extra input
uv run --no-editable --extra input openmle-task prepare-input \
  --task-dir /absolute/host/path/tasks/titanic \
  --sandbox-task-dir /mnt/tasks/titanic \
  --output artifacts/titanic.parquet
```

The first path is read by the preparation command. The second is recorded as
`metadata.data_dir` and must be accessible to the sandbox controller and workers.
See the [mount instructions](../openmle-sandbox/README.md#connect-the-sandbox-to-openmle-evo-and-openmle-erl).
Preparation does not copy data into the sandbox.

The output has one row containing the bundled original `prompt` and the package's
`metadata`. Private answers remain in the package; they are not copied into the
prompt. Use an absolute Parquet path in the consumer configuration because its
launcher can change the working directory.

## Prepare multiple tasks

Run the command once per selected package with a distinct output file. Combine
only the intended task inputs, listing their paths explicitly:

```bash
uv run --no-editable --extra input python - <<'PY'
from pathlib import Path
import pandas as pd

inputs = [Path("artifacts/titanic.parquet"), Path("artifacts/another-task.parquet")]
table = pd.concat([pd.read_parquet(path) for path in inputs], ignore_index=True)
table.to_parquet("artifacts/tasks.parquet", index=False)
print(f"Wrote {len(table)} tasks to artifacts/tasks.parquet")
PY
```

Replace the example list with files you have prepared. Prepare training and
evaluation task selections separately when they must be independent; this
command does not select tasks or generate data splits.

## Configure the consumer

| Consumer | Parquet setting | Final-scoring package parent |
| --- | --- | --- |
| Evo | `OPENMLE_EVAL_DATA` | `OPENMLE_SUBMIT_DATA_DIR_ROOT=/mnt/tasks` |
| Parallel SFT trajectory generation | `OPENMLE_PARALLEL_DATA` | Uses each record's `metadata.data_dir` |
| Evolutionary SFT trajectory generation | `OPENMLE_EVOLUTIONARY_DATA` | `OPENMLE_TASK_DATA_ROOT=/mnt/tasks` |
| RL | `PROMPT_DATA` for training, `EVAL_PROMPT_DATA` for evaluation | Uses each record's `metadata.data_dir` |

For the two evolutionary consumers, the parent plus the task directory name
resolves to `/mnt/tasks/titanic`. Their environment variable names differ.
Reusing the same package for search and final scoring reuses the same answers;
it is not an independent held-out evaluation.

Continue with the consumer's installation, model and sandbox configuration:

- [Evo](../../OpenMLE-Evo/docs/gym-task-packages.md#run-evo)
- [SFT trajectory generation](../../OpenMLE-ERL/SFT/docs/usage.md#generate-rollouts)
- [RL](../../OpenMLE-ERL/RL/docs/usage.md#configure)

SFT training itself consumes generated `id`/`messages` records, not these task
inputs. Generate trajectories and follow the SFT selection steps before training.
RL also requires its documented model, runtime and leaderboard configuration;
creating the input table does not launch a training job.
