"""Offline report plots for the iStratDE comparison (EXP-008, FR7).

Renders, from the already-built full report (actual persisted data only):

- ``paired-utility.png``: the paired per-seed utilities of the three
  methods within each task (primary ISTRATDE-RANDOM and secondary
  ISTRATDE-FIXED pairing visible); unobserved cells are gaps, never
  zero-filled.
- ``incumbent-vs-requests.png`` / ``incumbent-vs-tokens.png``:
  validation-incumbent versus ACTUAL cumulative ledger requests / charged
  tokens curves per cell. These are VALIDATION curves; no final-test
  feedback or search curves are ever plotted.

Pure offline rendering with the Agg backend: no network, no display, no
mutation of experiment artifacts. Empty/incomplete data renders as
explicitly annotated empty panels, not fabricated curves.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from research.comparison.protocol import canonical_protocol


class ComparisonPlotError(RuntimeError):
    """Plot rendering failed; the message IS the rule."""


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot

    return matplotlib.pyplot


def _render_paired(report: Mapping[str, Any], output_dir: Path) -> str:
    """Paired per-seed utility by method, one panel per task."""
    plt = _pyplot()
    proto = canonical_protocol()
    rows = report["matrix"]["rows"]
    fig, axes = plt.subplots(
        1, len(proto.tasks), figsize=(4.5 * len(proto.tasks), 4), sharey=True
    )
    if len(proto.tasks) == 1:
        axes = [axes]
    methods = list(proto.methods)
    x_positions = list(range(len(methods)))
    for axis, task in zip(axes, proto.tasks):
        task_rows = [row for row in rows if row["task_id"] == task.task_id]
        observed = 0
        for seed in proto.search_seeds:
            utilities = []
            for method in methods:
                row = next(
                    (
                        r
                        for r in task_rows
                        if r["method"] == method and r["search_seed"] == seed
                    ),
                    None,
                )
                utilities.append(
                    row["utility"] if row is not None else None
                )
            xs = [x for x, u in zip(x_positions, utilities) if u is not None]
            ys = [u for u in utilities if u is not None]
            observed += len(ys)
            if ys:
                axis.plot(xs, ys, "o-", label=f"seed {seed}")
        axis.set_xticks(x_positions, methods)
        axis.set_title(f"{task.task_id} (UCI {task.uci_id})")
        axis.set_xlabel("method")
        if not observed:
            axis.text(
                0.5,
                0.5,
                "no observed data",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
    axes[0].set_ylabel("utility (final accuracy; 0 = observed failure)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(loc="lower right", fontsize="small")
    fig.suptitle("Paired per-seed utility (conditional on the fixed test folds)")
    fig.tight_layout()
    path = output_dir / "paired-utility.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def _render_incumbent_curves(
    report: Mapping[str, Any],
    output_dir: Path,
    *,
    x_field: str,
    x_label: str,
    filename: str,
) -> str:
    """Validation incumbent vs actual cumulative cost, task x seed grid."""
    plt = _pyplot()
    proto = canonical_protocol()
    rows = report["matrix"]["rows"]
    fig, axes = plt.subplots(
        len(proto.search_seeds),
        len(proto.tasks),
        figsize=(4.5 * len(proto.tasks), 3.2 * len(proto.search_seeds)),
        sharex=False,
        sharey=True,
    )
    method_color = {"FIXED": "tab:gray", "RANDOM": "tab:blue", "ISTRATDE": "tab:red"}
    for row_axis, seed in zip(axes, proto.search_seeds):
        for axis, task in zip(row_axis, proto.tasks):
            cell_rows = [
                row
                for row in rows
                if row["task_id"] == task.task_id and row["search_seed"] == seed
            ]
            any_points = False
            for cell_row in cell_rows:
                curve = [
                    point
                    for point in cell_row["incumbent_curve"]
                    if point["validation_incumbent"] is not None
                ]
                if not curve:
                    continue
                any_points = True
                axis.step(
                    [point[x_field] for point in curve],
                    [point["validation_incumbent"] for point in curve],
                    where="post",
                    label=cell_row["method"],
                    color=method_color.get(cell_row["method"]),
                )
            axis.set_title(f"{task.task_id} seed {seed}", fontsize="small")
            axis.set_xlabel(x_label, fontsize="small")
            if not any_points:
                axis.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize="small",
                )
    for axis in axes[:, 0]:
        axis.set_ylabel("validation incumbent", fontsize="small")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right")
    fig.suptitle(
        "Validation incumbent vs "
        + x_label
        + " (ledger-derived; no final-test curves)"
    )
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def render_report_plots(
    report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Render all report figures; returns {filename: sha256}."""
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ComparisonPlotError("plot_output_symlinked")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        _render_paired(report, output_dir),
        _render_incumbent_curves(
            report,
            output_dir,
            x_field="cumulative_requests",
            x_label="cumulative actual requests",
            filename="incumbent-vs-requests.png",
        ),
        _render_incumbent_curves(
            report,
            output_dir,
            x_field="cumulative_charged_tokens",
            x_label=(
                "cumulative charged tokens "
                "(unknown usage retained at full reservation)"
            ),
            filename="incumbent-vs-tokens.png",
        ),
    ]
    return {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in names
    }
