"""Recheck and export aggregate calibration results after execution stops.

No model calls, sandbox submissions, or final-test data access. Raw ledgers,
candidate programs, model messages and prediction rows are not exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def generate(root, output):
    from research.calibration.collect import validate_trial
    from research.calibration.gates import read_json
    from research.calibration.plan import analyze
    from research.calibration.prepare import historical_integrity
    from research.calibration.specs import LEDGER, RUNTIME, selected_plan, sha256_file, trials
    runtime = root / RUNTIME
    outcome_path = runtime / 'outcome.json'
    outcome = read_json(outcome_path)
    manifest = read_json(runtime / 'manifest.json')
    ledger_path = root / LEDGER
    ledger_hash_before = sha256_file(ledger_path)
    outcome_hash_before = sha256_file(outcome_path)
    state = read_json(ledger_path)
    snapshot = outcome['budget']
    assert outcome['manifest_identity'] == manifest['identity'] == state['manifest_identity']
    assert snapshot['requests'] == state['requests']
    assert snapshot['started_at'] == state['started_at'] and snapshot['stopped'] == state['stopped']
    assert len({record['request_id'] for record in state['requests']}) == len(state['requests'])
    assert not any(record['status'] == 'reserved' for record in state['requests'])
    assert snapshot['global']['committed'] == {
        'requests': len(state['requests']),
        'tokens': sum(record['charged_tokens'] for record in state['requests'])}
    for record in state['requests']:
        if record['status'] == 'known':
            assert record['charged_tokens'] == sum(record['usage'].values())
        else:
            assert record['status'] == 'unknown' and record['charged_tokens'] == record['reserved_tokens']
    plan = selected_plan()
    analysis = analyze(plan, outcome['outcomes'])
    assert analysis == outcome['analysis'], 'analysis no longer matches frozen rule or integrity verdict'
    roster = trials(root, manifest['source_commit'])
    previous = {entry['normalized']['trial_id']: entry for entry in outcome['verified_trials']}
    verified = {}
    for trial in roster:
        if trial.trial_id in previous:
            current = validate_trial(root, manifest, trial, snapshot)
            assert current == previous[trial.trial_id]
            verified[trial.trial_id] = current
    preflight = read_json(runtime / 'preflight.json')
    assert historical_integrity(root) == preflight['historical']
    assert outcome['historical_artifacts_unchanged'] and outcome['stack_idle']
    assert sha256_file(ledger_path) == ledger_hash_before
    assert sha256_file(outcome_path) == outcome_hash_before
    independent = {'complete_matrix': not analysis['incomplete_trials']}
    if not analysis['incomplete_trials']:
        import numpy as np
        from scipy.stats import spearmanr
        values = np.array([row['repeat_values'] for row in analysis['configuration_summaries']])
        y = values.reshape(-1)
        # Independent least-squares projections rather than the analyzer's
        # direct row/column mean-square calculation.
        block = np.tile(np.eye(3)[:, 1:], (9, 1))
        configuration = np.repeat(np.eye(9)[:, 1:], 3, axis=0)
        x0 = np.column_stack([np.ones(27), block])
        x1 = np.column_stack([x0, configuration])
        rss0 = float(np.sum((y - x0 @ np.linalg.lstsq(x0, y, rcond=None)[0]) ** 2))
        rss1 = float(np.sum((y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]) ** 2))
        mse, msc = rss1 / 16, max(0.0, rss0 - rss1) / 8
        variance = max(0.0, (msc - mse) / 3)
        fraction = variance / (variance + mse) if variance + mse > 1e-25 else None
        correlations = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            if len(set(values[:, first])) == 1 or len(set(values[:, second])) == 1:
                correlations.append(None)
            else:
                correlations.append(float(spearmanr(values[:, first], values[:, second]).statistic))
        for field, value in (('ms_configuration', msc), ('ms_residual', mse),
                             ('estimated_configuration_variance', variance)):
            assert abs(analysis['variance'][field] - value) < 1e-12
        if fraction is not None:
            assert abs(analysis['variance']['estimated_configuration_variance_fraction'] - fraction) < 1e-10
        for actual, expected in zip(correlations, analysis['rank_agreement']['pairwise_spearman']):
            assert (actual is None and expected is None) or (actual is not None and expected is not None and abs(actual - expected) < 1e-12)
        independent.update({'least_squares_ms_configuration': msc, 'least_squares_ms_residual': mse,
                            'configuration_variance_fraction': fraction, 'scipy_pairwise_spearman': correlations,
                            'matches_frozen_analyzer': True})
    normalized = {row['trial_id']: row for row in outcome['outcomes']}
    rows, behavior_rows = [], []
    for trial in roster:
        data = normalized.get(trial.trial_id, {'status': 'not_run', 'validation_accuracy': None,
            'requests': 0, 'charged_tokens': 0, 'reason': 'not_run'})
        cell = snapshot['cells'][trial.trial_id]
        rows.append({'trial_id': trial.trial_id, 'repeat': trial.record['repeat'] + 1,
            'position': trial.record['position'] + 1, 'config_id': trial.record['config_id'],
            'inner_seed': trial.inner_seed, **trial.decoded,
            **{key: data[key] for key in ('status', 'validation_accuracy', 'requests', 'charged_tokens', 'reason')}})
        if trial.trial_id in verified:
            evidence = verified[trial.trial_id]
            behavior = evidence['behavior']
            selections = behavior['parent_selections']
            behavior_rows.append({'trial_id': trial.trial_id, 'config_id': trial.record['config_id'],
                **behavior['operator_counts'], 'initial_draft_best': behavior['initial_draft_best'],
                'score_informative_parent_events': sum(event['score_component_informative'] for event in selections),
                'parent_events': len(selections),
                'mean_score_weight_first_draw_tv_0_to_2': statistics.mean(event['score_weight_first_draw_tv_0_to_2'] for event in selections) if selections else None,
                'result_sha256': evidence['result_sha256']})
    counts = Counter()
    job_counts = Counter()
    invalid_python = 0
    for evidence in verified.values():
        counts.update(evidence['behavior']['operator_counts'])
        job_counts.update(evidence['job_counts'])
        directory = runtime / 'runs' / evidence['normalized']['trial_id']
        nodes = [json.loads(line) for line in (directory / 'journal.jsonl').read_text().splitlines() if line.strip()]
        invalid_python += sum(node.get('is_buggy') is True and
            'empty_candidate_code:invalid_python' in str(node.get('term_out', '')) for node in nodes)
    behavior_by_id = {row['trial_id']: row for row in behavior_rows}
    crossover_summary = {}
    for probability in (0.05, 0.5, 0.95):
        subset = [behavior_by_id[row['trial_id']] for row in rows
                  if row['crossover_prob'] == probability and row['trial_id'] in behavior_by_id]
        crossover = sum(row['crossover'] for row in subset)
        decisions = crossover + sum(row['improve'] for row in subset)
        crossover_summary[str(probability)] = {'crossovers': crossover, 'improve_or_crossover': decisions,
            'observed_share': crossover / decisions if decisions else None}
    later_improvement_runs = sum(entry['normalized']['validation_accuracy'] is not None and
        entry['behavior']['initial_draft_best'] is not None and
        entry['normalized']['validation_accuracy'] > entry['behavior']['initial_draft_best']
        for entry in verified.values())
    result = {'schema': 'calibration-public-report.v1', 'status': outcome['status'],
        'source_commit': manifest['source_commit'], 'manifest_identity': manifest['identity'],
        'protocol': manifest['protocol'], 'plan_identity': plan['plan_identity'],
        'started_at': outcome['started_at'], 'finished_at': outcome['finished_at'],
        'elapsed_seconds': outcome['elapsed_seconds'], 'stopped_reason': outcome['stopped_reason'],
        'cost': snapshot['global']['committed'],
        'unknown_usage_requests': sum(record['status'] == 'unknown' for record in state['requests']),
        'analysis': analysis, 'independent_recomputation': independent,
        'verified_trials': len(verified), 'independently_rescored_jobs': sum(len(e['rescored_jobs']) for e in verified.values()),
        'operator_counts': dict(counts), 'job_counts': dict(job_counts),
        'pre_submission_invalid_python_candidates': invalid_python,
        'mechanism_summary': {'crossover_by_configured_probability': crossover_summary,
            'runs_beating_initial_drafts': later_improvement_runs,
            'parent_selection_events': sum(row['parent_events'] for row in behavior_rows),
            'score_informative_parent_events': sum(row['score_informative_parent_events'] for row in behavior_rows)},
        'fixed_preflight_jobs': 1, 'new_final_evaluations': 0,
        'old_comparison_files_unchanged': preflight['historical']['original_comparison_files_verified'],
        'historic_ledgers_and_final_only_stage_unchanged': True, 'cleanup_verified': True,
        'private_outcome_sha256': outcome_hash_before, 'private_ledger_sha256': ledger_hash_before}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    for name, records in (('trials-27.csv', rows), ('operator-behavior.csv', behavior_rows)):
        if records:
            with (output / name).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(records)
    (output / 'protocol.json').write_text(json.dumps(manifest['protocol'], indent=2) + '\n')
    (output / 'schedule.json').write_text(json.dumps(plan, indent=2) + '\n')
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, rows = generate(args.repo_root.resolve(), args.output)
    render(args.output, result, rows)
    print(json.dumps({'status': result['status'], 'verdict': result['analysis']['verdict'],
                      'cost': result['cost'], 'output': str(args.output)}))


# Rendering is defined below so collection can be imported independently.
def render(output, result, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    analysis = result['analysis']
    summaries = analysis['configuration_summaries']
    parameters = {row['config_id']: row for row in rows}
    colors = ['#24699A', '#CB7E2D', '#458C75']
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for repeat in range(3):
        xs, ys = [], []
        for index, row in enumerate(summaries):
            value = row['repeat_values'][repeat]
            if value is not None:
                xs.append(index + (repeat - 1) * 0.13)
                ys.append(100 * value)
        ax.scatter(xs, ys, color=colors[repeat], s=42, label=f'Repetition {repeat + 1}', zorder=3)
    for index, row in enumerate(summaries):
        if row['mean'] is not None:
            ax.errorbar(index, row['mean'] * 100,
                yerr=(row['sample_sd'] or 0) * 100, fmt='D', color='#202020',
                markersize=4, capsize=4, linewidth=1.2,
                label='Mean ± 1 repeat SD' if index == 0 else None, zorder=4)
    labels = [f"{row['config_id']}\np={parameters[row['config_id']]['crossover_prob']:g}, w={parameters[row['config_id']]['score_weight']:g}" for row in summaries]
    ax.set_xticks(range(9), labels, fontsize=9)
    ax.set_ylabel('Best trusted validation accuracy (%)')
    ax.set_title('Spambase: nine configurations, three independent repetitions', loc='left', pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.text(0.08, 0.01, 'Error bars describe repeat variability, not confidence intervals. No final-test scores.', fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output / 'validation-repeatability.png', dpi=220)
    plt.close(fig)
    if analysis['variance'] is not None:
        fraction = analysis['variance']['estimated_configuration_variance_fraction']
        rank = analysis['rank_agreement']['median_spearman']
        fraction_text = f'{fraction:.4f}' if fraction is not None else '无可估计差异'
        rank_text = f'{rank:.4f}' if rank is not None else '至少一组排名恒定，无法判定'
    else:
        fraction_text = rank_text = '实验矩阵不完整，未计算稳定性结论'
    table = []
    for row in summaries:
        params = parameters[row['config_id']]
        mean = f"{row['mean'] * 100:.3f}%" if row['mean'] is not None else '缺失'
        sd = f"{row['sample_sd'] * 100:.3f}" if row['sample_sd'] is not None else '—'
        delta = f"{row['mean_paired_difference_from_center'] * 100:+.3f}" if row['mean_paired_difference_from_center'] is not None else '—'
        table.append(f"| {row['config_id']} | {params['crossover_prob']:g} | {params['score_weight']:g} | {row['completed_observations']}/3 | {mean} | {sd} | {delta} |")
    write_readme(output, result, fraction_text, rank_text, table)


def write_readme(output, result, fraction, rank, table):
    verdict = result['analysis']['verdict']
    conclusions = {
        'exploratory_stable_configuration_signal': '本轮观察到符合预设门槛的配置稳定性信号。',
        'stable_configuration_signal_not_demonstrated': '本轮未达到预设的配置稳定性门槛。',
        'insufficient_complete_evidence': '实验矩阵不完整，无法按预设规则判断配置稳定性。',
    }
    conclusion = conclusions.get(verdict, '存在尚未解决的完整性或清理问题，不能作稳定性结论。')
    jobs = result['job_counts']
    usage_text = ('全部请求均有提供方用量回报，没有未结算请求。' if result['unknown_usage_requests'] == 0
                  else '存在未知用量，已按完整预留额度记账并停止；该额度不是已确认实际消耗。')
    shares = result['mechanism_summary']['crossover_by_configured_probability']
    behavior_text = '、'.join(f"p={key} 时 {value['crossovers']}/{value['improve_or_crossover']}" for key, value in shares.items())
    recomputation_text = ('完整矩阵的方差分解另以最小二乘投影复算，排名相关另以 SciPy 复算。'
        if result['independent_recomputation']['complete_matrix'] else
        '由于矩阵不完整，没有进行完整矩阵的方差及排名稳定性复算。')
    lines = [
        '# Spambase 两维重复性校准结果', '', '**' + conclusion + '**', '',
        '本轮只检验交叉概率与成绩权重能否在固定内层搜索中产生可重复的验证表现差异。没有运行外层 iStratDE，也没有新增最终测试评测。', '',
        f"共核验 {result['verified_trials']}/27 次独立搜索；用量为 {result['cost']['requests']} 次模型请求、{result['cost']['tokens']:,} tokens，耗时 {result['elapsed_seconds']/60:.2f} 分钟。批准上限为 216 次请求、600,000 tokens、4 小时。",
        f"候选作业提交 {jobs.get('submitted', 0)} 次，成功评分 {jobs.get('scored', 0)} 次，执行失败 {jobs.get('execution_failed', 0)} 次；失败请求的费用也计入用量。另有 1 次固定隔离探针，模型请求为 0，不计入实验样本。", '',
        f"另有 {result['pre_submission_invalid_python_candidates']} 个候选因 Python 语法无效未提交作业，其模型费用同样计入上述请求及 tokens。", '',
        '## 预先规定的稳定性判断', '',
        f'- 去除重复批次整体偏移后的估计配置方差占比：**{fraction}**；门槛为 ≥ 0.5。',
        f'- 三个重复批次两两配置排名的 Spearman 相关系数中位数：**{rank}**；门槛为 ≥ 0.5。',
        '- 必须完成全部九配置 × 三重复，并同时达到两个门槛；缺失值不填补，不追加试验。', '',
        '这里的方差占比是本设计下的探索性描述量，草案将其称为估计 ICC。它不是总体参数或显著性检验；三次重复不足以证明某配置普遍更优。', '',
        '## 九组配置', '',
        '其余参数固定：进步权重 0.4、新颖性权重 0.25、调试深度 1。每次搜索三代、每代两个主候选，最多两次调试请求。中心配置 c04 为交叉概率 0.5、成绩权重 1。', '',
        '| 配置 | 交叉概率 | 成绩权重 | 有效重复 | 平均最佳验证准确率 | 重复标准差（百分点） | 相对中心的平均配对差（百分点） |',
        '|---|---:|---:|---:|---:|---:|---:|', *table, '',
        '![九组配置的三次独立重复](validation-repeatability.png)', '',
        '彩色点为单次搜索最佳可信验证准确率；黑色标记为均值，误差线为重复标准差，并非置信区间。', '',
        '## 实际搜索行为', '',
        '交叉概率确实改变了操作选择。实际交叉次数／交叉或改进操作总次数为：' + behavior_text + '。',
        f"{result['mechanism_summary']['runs_beating_initial_drafts']}/27 次搜索的后续候选超过初始两份草稿的最佳验证分数；{result['mechanism_summary']['score_informative_parent_events']}/{result['mechanism_summary']['parent_selection_events']} 次父代选择中成绩分量有区分度。以上均为已记录轨迹的描述，未与同预算纯草稿生成作因果对照。", '',
        '## 核验与研究边界', '',
        f"每次收录结果均重新核对请求与账本、操作参数、候选源码、后台预测快照和验证答案哈希；共独立复算 {result['independently_rescored_jobs']} 份可信验证预测。" + recomputation_text,
        f"原比较实验的 {result['old_comparison_files_unchanged']:,} 个文件、历史账本和最终评测补齐阶段保持不变；本轮结束后确认栈空闲。未知用量请求数：{result['unknown_usage_requests']}。" + usage_text, '',
        '重复批次内共用搜索随机种子，九组配置的执行顺序预先随机化；模型输出仍有随机性。结果条件于这一份固定训练／验证拆分、该模型服务和当前短搜索设置。排名稳定性不是外层优化器优越性，也不是独立泛化证明。', '',
        '- [逐次结果与成本](trials-27.csv)', '- [操作与父代选择敏感性](operator-behavior.csv)',
        '- [机器报告及独立复算](report.json)', '- [已选协议](protocol.json)、[冻结顺序](schedule.json)', '',
        '顺序文件保留了生成草案时的准备状态标记；本轮实际授权由独立私有凭证绑定实验清单，不能从公开顺序文件推断新的执行授权。', '',
        '父代敏感性仅在实际记录的候选池上改变成绩权重，复算首个父代抽样概率；不代表新配置会产生多少准确率提升。原始模型消息、候选程序、预测行、账本及授权凭证保留在私有运行目录，不随报告发布。', '',
        f"执行源码：`{result['source_commit']}`；实验清单标识：`{result['manifest_identity']}`。", '',
    ]
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
