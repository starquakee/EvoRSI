# Task input example

`titanic-extended-task.json` contains a complete `prompt`/`metadata` record for
the `titanic-extended@1` task. Download it from `FrontisAI/OpenMLE-Tasks` using
task key `smith_tasks.txt:2968`.

Use `openmle-task prepare-input` to prepare a Parquet from the downloaded
package. Set `--sandbox-task-dir` to the task root visible to the sandbox
controller and workers. To reuse the example record's identity, set:

```bash
--task-uuid 340c9aa6-df8a-43d5-899b-846ed577cb7b
```

The record contains:

- `prompt`: system and user messages from the SFT4 selfvalid0327 template.
- `metadata.data_dir`: a placeholder for the sandbox-visible task path.
- `metadata.task_description`: the public task description.
- `metadata.data_description`: the package's data preview.
- Task identity, resource type, metric direction and score bounds.

The reference record comes from
`automl_parquet_sft_4_1643_selfvalid0327_filtered/train.parquet`.
The template is packaged at `openmle_gym/templates/selfvalid0327-sft4.json`.

See [download and run instructions](../../../OpenMLE-Evo/docs/gym-task-packages.md).
