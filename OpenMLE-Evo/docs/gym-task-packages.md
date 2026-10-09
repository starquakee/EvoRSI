# Run a Gym task with OpenMLE-Evo

A downloaded Task package is the data used by the sandbox. Evo reads an evaluation
Parquet with `prompt` and `metadata`, then sends `metadata.data_dir` to that
sandbox. Downloading a package does not, by itself, create this evaluation input.

The workflow is:

```text
openmle-task prepare-input (Gym data preparation)
  -> eval.parquet
  -> scripts/evaluate_airaevo.py
  -> third_party/aira-evo/examples/mle_bench/build_tasks.py
  -> SandboxMLEBenchTask.evaluate_code()
  -> sandbox /api/v1/jobs
  -> data/public + data/private/test_answer.csv + utils/metric.py
```

This example uses `experiment/openmle_evo_smoke`.

## Which path goes where?

Suppose the complete package is visible to the sandbox controller and workers as:

```text
/mnt/pubdatasets2/tasks/titanic-extended@1/
├── data/public/
├── data/private/test_answer.csv
└── utils/metric.py
```

| Setting | Value | Meaning |
| --- | --- | --- |
| Parquet `metadata.data_dir` | `/mnt/pubdatasets2/tasks/titanic-extended@1` | Single task-package root used during search |
| `OPENMLE_SUBMIT_DATA_DIR_ROOT` | `/mnt/pubdatasets2/tasks` | Parent of task packages used for final scoring |
| Resolved final task path | `/mnt/pubdatasets2/tasks/titanic-extended@1` | Submit root plus the basename of `metadata.data_dir` |

Do not set the submit root to the single task directory: the existing adapter
appends its basename, which would produce `.../titanic-extended@1/titanic-extended@1`.
Neither setting points directly to `data/public` or `data/private`.
Set `OPENMLE_SUBMIT_DATA_DIR_ROOT` explicitly to avoid the older
`OPENMLE_SUBMIT_DIR` fallback and its directory-name replacement in generated code.

The dispatcher exports `DATA_DIR=<package>/data/public` for candidate code and
scores the resulting `submission.csv` using `<package>/data/private/test_answer.csv`
and `<package>/utils/metric.py`. The filename `test_answer.csv` does not select
the Evo phase. The selected package path does.

These are **container-visible paths**, not necessarily the paths on the download
machine. The sandbox/router does not transfer task data. See the
[sandbox mount instructions](../../OpenMLE-Gym/openmle-sandbox/README.md#connect-the-sandbox-to-openmle-evo-and-openmle-erl)
and ensure both the controller and workers can resolve the same package path.

## Download and prepare a task

The dataset index distinguishes two release types: `built_task` contains a
complete task package, while `recipe` contains preparation code for upstream
data. Both produce the same task-package layout for Evo.

Download the launcher and index, and choose where to store task packages:

```bash
TASKS_ROOT=/absolute/host/path/tasks
uvx --from 'huggingface_hub>=0.34' hf download FrontisAI/OpenMLE-Tasks \
  build_task.py task_index.jsonl requirements.txt \
  --repo-type dataset --local-dir OpenMLE-Tasks
cd OpenMLE-Tasks
```

Choose one of the following routes.

### Download a complete package

```bash
TASK_NAME=titanic-extended@1
uv run --with 'huggingface_hub>=0.34' --with 'zstandard>=0.23' \
  python build_task.py --task-key 'smith_tasks.txt:2968' \
  --output-dir "$TASKS_ROOT/$TASK_NAME"
```

### Build a package from downloaded upstream data

For a `recipe` entry, open the `source_urls` listed in `task_index.jsonl`, sign in
to the upstream platform and accept its terms. Download the required raw files
from that platform. The `--accept-upstream-terms` flag confirms you have completed
that step; it does not accept terms or download gated files on your behalf.

For the Titanic recipe, obtain the competition CSV files from
[Kaggle](https://www.kaggle.com/competitions/titanic/data) and extract them into
`/absolute/path/to/titanic-raw`. Then download the recipe and build the package:

```bash
TASK_NAME=titanic
uvx --from 'huggingface_hub>=0.34' hf download FrontisAI/OpenMLE-Tasks \
  --repo-type dataset --include 'recipes/titanic/paper_tasks.txt--156/**' \
  --local-dir .
uv run --with pandas --with scikit-learn python build_task.py \
  --task-key 'paper_tasks.txt:156' \
  --raw-dir /absolute/path/to/titanic-raw \
  --accept-upstream-terms \
  --output-dir "$TASKS_ROOT/$TASK_NAME"
```

The launcher uses a copy of the supplied raw directory. The resulting package
contains public training/test files, private answers, the metric and task metadata.
Pass this generated package to Evo, rather than the raw download or recipe folder.

### Prepare the Evo input

Make the generated package available to the sandbox at
`/mnt/pubdatasets2/tasks/$TASK_NAME`.
Clone the code if you have not already done so:

```bash
git clone https://github.com/FrontisAI/OpenRSI.git
cd OpenRSI/OpenMLE-Gym
uv sync --no-editable --extra input
uv run --no-editable --extra input openmle-task prepare-input \
  --task-dir "$TASKS_ROOT/$TASK_NAME" \
  --sandbox-task-dir "/mnt/pubdatasets2/tasks/$TASK_NAME" \
  --output ../OpenMLE-Evo/artifacts/gym-example/eval.parquet
cd ../OpenMLE-Evo
mkdir -p artifacts/gym-example/leaderboards
```

Install Evo using its [installation instructions](usage.md#2-installation).

The Gym command reads the package's `info/task_metadata.json`,
`info/data_description.txt`, and `data/public/description.txt`. Download the
complete archive, including `info/`; it cannot prepare an input from a partial
listing with those files omitted. It copies the existing metadata fields and
descriptions and binds `metadata.data_dir` to the sandbox-visible task path.
It preserves a package UUID when available, otherwise creates an input UUID;
`--task-uuid` can reuse an existing record's identity.

The prompt template is `selfvalid0327-sft4.json`. It specifies the
available libraries, resource assumptions, submission format and validation-score
output. Configure the sandbox resources to match the template.

## Run Evo

Configure model and sandbox access using `.env.example`, then run:

```bash
export OPENMLE_EVAL_DATA="$PWD/artifacts/gym-example/eval.parquet"
export OPENMLE_LEADERBOARD_DIR="$PWD/artifacts/gym-example/leaderboards"
export OPENMLE_SUBMIT_DATA_DIR_ROOT=/mnt/pubdatasets2/tasks
OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke ./scripts/run_standard.sh
```

The commands above use an empty leaderboard directory, so they can run without
downloaded leaderboard CSVs. To report leaderboard medals, provide the task's
leaderboard files through `OPENMLE_LEADERBOARD_DIR`.

Inspect `program_ep_*/<task-name>/stat.json`, the generated code/logs,
and the final `submit_score`. A prepared Parquet alone is not an executed run.

## One package versus independent validation and test packages

Both phases may point to the same package when trying the workflow. This performs
search and final re-evaluation against the same answer set; the final score is
**not an independent held-out test result**. The existing default uses sandbox
scores for search selection (`trust_model_validation_score=false`).

For an independently prepared test package, use matching task directory names
under separate parents:

```text
metadata.data_dir = /mnt/pubdatasets2/validation/titanic-extended@1
OPENMLE_SUBMIT_DATA_DIR_ROOT=/mnt/pubdatasets2/test
resolved final path = /mnt/pubdatasets2/test/titanic-extended@1
```

Each package must have its own public input files, private answers and evaluator
in the standard layout. Duplicating one package under two names does not create
independent data splits. Evo does not infer or generate these splits from the
presence of `test_answer.csv`.

## Other consumers

See [Prepare task inputs](../../OpenMLE-Gym/docs/task-inputs.md) for multi-task
tables and consumer-specific package paths. Evolutionary SFT uses
`OPENMLE_TASK_DATA_ROOT` for its final-scoring parent; Evo uses
`OPENMLE_SUBMIT_DATA_DIR_ROOT`.

The Parquet contains `prompt` and `metadata` columns.

| Consumer | Input setting | Reader |
| --- | --- | --- |
| Evo | `OPENMLE_EVAL_DATA` | `OpenMLE-Evo/scripts/evaluate_airaevo.py` |
| Parallel SFT trajectory generation | `OPENMLE_PARALLEL_DATA` | `OpenMLE-ERL/SFT/tts_search/evaluate_pass_k.py` |
| Evolutionary SFT trajectory generation | `OPENMLE_EVOLUTIONARY_DATA` | `OpenMLE-ERL/SFT/third_party/aira-evo/examples/mle_bench/build_tasks.py` |
| RL rollout | `PROMPT_DATA` / `EVAL_PROMPT_DATA` | SLIME prompt loader and `OpenMLE-ERL/RL/generate_mle.py` |

These consumers retain their own documented execution and scoring configuration.
An SFT **training** table contains generated `messages`; it is not the task input
table above. See the [SFT guide](../../OpenMLE-ERL/SFT/docs/usage.md#input-formats)
and [RL guide](../../OpenMLE-ERL/RL/docs/usage.md).

See the [sample input](../../OpenMLE-Gym/examples/consumer-inputs/README.md) for a complete record.
