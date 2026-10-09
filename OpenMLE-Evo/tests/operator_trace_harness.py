"""Shared harness for US-007 operator-trace tests (not a test module).

Builds a REAL ``Evolutionary`` solver with a complete solver-config mapping
(every ``EvolutionarySolverConfig`` field the solver reads; a plain DictConfig
because the upstream base dataclass declares ``field(default_factory={})``,
which OmegaConf structured configs reject) in generation mode, with two
documented monkeypatch hook points:

1. ``dojo.solvers.evo.evo.GenericLLM`` is replaced with a deterministic fake
   whose ``__call__`` returns content that the REAL operators
   (``draft_op``/``improve_op``/``debug_op``/``crossover_op`` in
   ``dojo/core/solvers/operators/``) parse into code via
   ``execute_op_plan_code`` -> ``extract_code``: a fenced ```python block
   containing valid Python. Returned code embeds a deterministic marker
   (operator name + per-instance call counter) so child hashes differ.
2. The task object is a fake: ``step_task(state, code)`` returns a scripted
   ``eval_result`` with deterministic ``VALIDATION_FITNESS`` derived from the
   code string (``SCORE = <float>`` marker; ``FAIL`` marker => buggy), so
   ``parse_eval_result`` takes the deterministic-metadata path and never
   calls the analyze LLM.

Both ``random`` and ``numpy.random`` are seeded in ``build_solver``.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy
from omegaconf import OmegaConf

from dojo.core.interpreters.base import ExecutionResult
from dojo.core.tasks.constants import (
    AUX_EVAL_INFO,
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    VALID_SOLUTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
)
from dojo.solvers.evo.evo import Evolutionary
from dojo.utils.logger import config_logger

OPERATOR_NAMES = ("draft", "improve", "debug", "crossover", "analyze")

TASK_INFO = {
    TASK_DESCRIPTION: "Predict the target column as well as possible.",
    "lower_is_better": False,
}

DEFAULT_SEED = 1234


def _operator_cfg_dict(name: str) -> Dict[str, Any]:
    template = {"template": "", "input_variables": [], "partial_variables": {}}
    return {
        "llm": {
            "client": {
                "api": "litellm",
                "model_id": f"fake-{name}",
                "base_url": "http://example.invalid",
                "api_key": "fake-key",
                "use_azure_client": False,
                "provider": "openai",
            },
            "generation_kwargs": {},
        },
        "system_message_prompt_template": dict(template),
        "init_user_message_prompt_template": dict(template),
        "user_message_prompt_template": dict(template),
    }


def make_solver_config(checkpoint_dir: Path, overrides: Optional[Dict[str, Any]] = None):
    """Config mapping with every EvolutionarySolverConfig field the solver reads.

    Note: ``OmegaConf.structured(EvolutionarySolverConfig`` cannot be used
    because the upstream base SolverConfig declares
    ``operators: dict = field(default_factory={})`` (a dict instance, not a
    callable), which OmegaConf rejects. A plain DictConfig carries the same
    fields; the solver only uses attribute/item access.
    """
    base: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_dir),
        "step_limit": 12,
        "time_limit_secs": 3600,
        "execution_timeout": 600,
        "num_islands": 1,
        "max_island_size": 50,
        "crossover_prob": 0.5,
        "migration_prob": 0.0,
        "initial_temp": 1.0,
        "final_temp": 1.0,
        "num_generations_till_migration": 999,
        "num_generations_till_crossover": 1,
        "num_generations": 3,
        "individuals_per_generation": 2,
        "max_debug_depth": 1,
        "max_debug_time": 600,
        "fresh_draft_prob": 0.0,
        "data_preview": False,
        "use_test_score": False,
        "use_complexity": False,
        "export_search_results": False,
        "execution_mode": "generation",
        "max_llm_call_retries": 3,
        "available_packages": ["numpy", "pandas", "scikit-learn"],
        "memory": {"memory_processor": "simple_memory", "memory_op_kwargs": {}},
        "debug_memory": {"memory_processor": "simple_memory", "memory_op_kwargs": {}},
        "few_shot": {"improve": 1, "crossover": 2},
        "experience": {
            "enabled": True,
            "prompt_memory": {"enabled": False},
        },
        "operators": {name: _operator_cfg_dict(name) for name in OPERATOR_NAMES},
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def default_responder(operator: str, call_index: int, query_data: Dict[str, Any]) -> str:
    """Deterministic fake-LLM completion parseable by the real operators."""
    marker = f"{operator}-{call_index}"
    code = (
        "import pandas as pd\n"
        "SCORE = 0.5\n"
        f"MARKER = \"{marker}\"\n"
    )
    return f"Plan for {marker}.\n```python\n{code}```\n"


def score_from_code(code: str) -> Optional[float]:
    match = re.search(r"^SCORE\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$", code, re.MULTILINE)
    return float(match.group(1)) if match else None


def success_result(score: float, feedback: str = "ok") -> Dict[str, Any]:
    return {
        EXECUTION_OUTPUT: ExecutionResult(
            term_out=[feedback], exec_time=0.01, exit_code=0
        ),
        VALIDATION_FITNESS: float(score),
        AUX_EVAL_INFO: {"status": "success", "feedback": feedback},
        VALID_SOLUTION: True,
        VALID_SOLUTION_FEEDBACK: "",
    }


def failure_result(feedback: str = "scripted failure") -> Dict[str, Any]:
    return {
        EXECUTION_OUTPUT: ExecutionResult(
            term_out=[feedback], exec_time=0.01, exit_code=1
        ),
        VALIDATION_FITNESS: None,
        AUX_EVAL_INFO: {"status": "failed", "feedback": feedback},
        VALID_SOLUTION: True,
        VALID_SOLUTION_FEEDBACK: "",
    }


def default_evaluator(code: str, eval_index: int) -> Dict[str, Any]:
    """Deterministic evaluator: FAIL marker or missing SCORE => buggy."""
    if "FAIL" in code:
        return failure_result()
    score = score_from_code(code)
    if score is None:
        return failure_result("no score marker")
    return success_result(score)


class FakeTask:
    """Fake task: everything the sync ``search()`` loop touches."""

    def __init__(self, evaluator: Optional[Callable[[str, int], Dict[str, Any]]] = None):
        self.evaluator = evaluator or default_evaluator
        self.stop_requested = False
        self.eval_count = 0
        self.codes: list[str] = []

    def step_task(self, state, code):
        self.eval_count += 1
        self.codes.append(code)
        return state, self.evaluator(code, self.eval_count)


def build_solver(
    tmp_path: Path,
    monkeypatch,
    *,
    responder: Optional[Callable[[str, int, Dict[str, Any]], str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    seed: int = DEFAULT_SEED,
):
    """Build a real Evolutionary with fake GenericLLMs patched in.

    Returns (solver, fake_llms) where fake_llms maps operator name -> fake
    GenericLLM instance (call_tracker counts LLM invocations).
    """
    random.seed(seed)
    numpy.random.seed(seed)
    config_logger(None)  # low-resource global logger: no file/wandb handlers
    responder = responder or default_responder
    fake_llms: Dict[str, Any] = {}

    class FakeGenericLLM:
        def __init__(self, cfg):
            model_id = str(cfg.llm.client.model_id)
            self.operator = model_id.removeprefix("fake-")
            self.call_tracker = 0
            fake_llms[self.operator] = self

        @property
        def client_content_key(self):
            return "content"

        def __call__(self, query_data=None, messages=None, **kwargs):
            self.call_tracker += 1
            text = responder(self.operator, self.call_tracker, dict(query_data or {}))
            usage = {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10,
                "cumulative_num_llm_calls": self.call_tracker,
            }
            return text, {
                "usage": usage,
                "prompt_messages": messages,
                "completion_text": text,
            }

    monkeypatch.setattr("dojo.solvers.evo.evo.GenericLLM", FakeGenericLLM)
    cfg = make_solver_config(Path(tmp_path) / "checkpoint", overrides)
    solver = Evolutionary(cfg, task_info=dict(TASK_INFO))
    return solver, fake_llms


def read_trace(path: Path) -> list[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
