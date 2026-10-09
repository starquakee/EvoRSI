# EEMA Skill synthesis materials

This directory contains the task descriptions, category playbooks, synthesis prompt, and task-specific skills used by the OpenMLE-Evo skill workflow.

## Repository Contents

| Path                                     | Purpose                                                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `system.txt`                             | Environment constraints for generated solutions.                                                                               |
| `descriptions/cleaned_<kaggle-slug>.txt` | Official task statements, evaluation metrics, submission contracts, and dataset descriptions.                                  |
| `playbooks/<modality>-<task-type>.md`    | Category-level modeling guidance organized by modality and task type.                                                          |
| `task_skill_synth_prompt.txt`            | Prompt used to combine one task description, the relevant playbooks, and the execution constraints into a task-specific skill. |
| `OUTPUTS/SKILL_<task>.md`                | Working output created by the synthesis prompt. The directory may be created locally when the prompt is run.                   |
| `task_skills/SKILL_<task>.md`            | Task-specific skills included with the repository. `SKILL_error.md` is the failure prevention skill.                           |

## Environment Configuration

`system.txt` should describe the user's actual execution environment rather than a generic or assumed setup. It should record the available hardware, operating constraints, installed libraries and versions, data and output locations, time and memory limits, and any benchmark-specific execution requirements. Update it whenever the environment changes so that generated skills recommend feasible models and workflows.
