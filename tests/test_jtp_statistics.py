import csv
import json
import numpy as np
import pytest
from evaluation.ablation.jtp.distributions import KEYS, FLAGS
from evaluation.ablation.jtp.statistics import step_records, report, write_reports
from evaluation.ablation.jtp.reporting import paper_table, _summary


def make_records():
    n, h = 2, 4
    values = {k: [np.zeros(h-1 if k in ('curvature', 'acceleration_rate', 'steering_rate') else h) for _ in range(n)] for k in KEYS}
    bad = {k: [v.astype(bool) for v in vs] for k, vs in values.items()}
    # Road violation at step 1 is excluded from step reports.
    bad['road'][0][0] = True
    # Different vehicles violate at different prediction steps.
    bad['curvature'][0][0] = True  # curvature belongs to step 2
    bad['yaw'][1][2] = True  # step 3
    meta = [dict(seed=0, episode_index=1, frame_index=2, ego_index=i) for i in range(n)]
    deviation = np.array([[100., 1., 2., 3.], [100., 4., 5., 6.]])
    records = []
    for m in ('Baseline', 'Ours'):
        records.extend(step_records((values, bad), meta, m, deviation))
    return records


def test_step_alignment_joint_and_single_vehicle():
    rows = make_records()
    summaries, _, confidence = report(rows, 'step', 20)
    for m in ('Baseline', 'Ours'):
        s = summaries[m]
        assert s['num_samples'] == 6
        assert s['num_joint_samples'] == 3
        assert s['joint_feasibility_pct'] == pytest.approx(100/3)
        assert s['feasibility_pct'] == pytest.approx(200/3)
        assert s['geometric_violation_pct'] == pytest.approx(100/6)
        assert s['stability_violation_pct'] == pytest.approx(100/6)
        assert s['road_violation_pct'] == 0
    assert summaries['Ours']['deviation_m'] == 3.5
    assert confidence['feasibility_pct']['bootstrap_95ci_pp'] == [0., 0.]
    assert all(r['prediction_step'] in (2, 3, 4) for r in rows)


def test_output_names_csv_recomputation_and_empty(tmp_path):
    rows = make_records()
    write_reports(tmp_path, rows, 'step', [0], 3, 2026, [])
    directory = tmp_path/'step'
    with (directory/'jtp_per_vehicle_step.csv').open(encoding='utf-8-sig') as f:
        saved = list(csv.DictReader(f))
    actual = np.mean([r['joint_feasible'] == 'True' for r in saved if r['method'] == 'Ours'])*100
    with (directory/'jtp_summary_step.csv').open(encoding='utf-8-sig') as f:
        assert float(list(csv.DictReader(f))[1]['feasibility_pct']) == actual
    assert not (tmp_path/'jtp_summary.csv').exists()
    table = (directory/'projection_ablation_table_step.tex').read_text()
    assert 'tab:projection_ablation_step' in table
    assert 'lccccc' in table
    empty, _, _ = report([], 'step')
    assert empty['Ours']['num_samples'] == 0
    assert empty['Ours']['joint_feasibility_pct'] is None
    assert 'N/A' in paper_table(empty, 'step')
    write_reports(tmp_path/'empty', [], 'step', [0], 3, 1, [])


def test_projection_timing_uses_individual_vehicle_calls(tmp_path):
    from evaluation.ablation.jtp.statistics import add_projection_timing
    rows = make_records()
    # Timing is sampled once per vehicle, not once per synchronized three-vehicle frame.
    small = [{**r, 'frame_index': 3} for r in rows if r['ego_index'] == 0 and r['prediction_step'] == 2]
    other_seed = [{**r, 'seed': 1} for r in rows]
    frames = [
        dict(seed=0, episode_index=1, frame_index=2,
             projection_latency_ms_by_vehicle=[10., 20., 30.]),
        dict(seed=0, episode_index=1, frame_index=3,
             projection_latency_ms_by_vehicle=[30., 40., 50.]),
        dict(seed=1, episode_index=1, frame_index=2,
             projection_latency_ms_by_vehicle=[80., 100., 120.]),
    ]
    result = write_reports(tmp_path, rows+small+other_seed, 'step', [0, 1], 0, 1, frames)
    ours = result['summary']['Ours']
    assert result['summary']['Baseline']['projection_latency_ms_mean'] == 0.
    assert ours['projection_latency_ms_mean'] == pytest.approx(480. / 9.)
    assert ours['projection_latency_ms_p95'] == pytest.approx(np.percentile(
        [10., 20., 30., 30., 40., 50., 80., 100., 120.], 95))
    assert ours['num_vehicle_projections'] == 9
    assert result['timing']['unit'] == 'ms_per_vehicle_projection'
    assert result['timing']['num_vehicle_projections'] == 9
    assert result['timing']['num_synchronized_frames'] == 3
    assert result['timing']['per_seed'] == {
        '0': {'Baseline': 0., 'Ours': 30.},
        '1': {'Baseline': 0., 'Ours': 100.},
    }
    table = (tmp_path/'step/projection_ablation_table_step.tex').read_text()
    assert 'Mean Time' in table and 'lcccccc' in table and '53.33' in table
    assert all(line.count('&') == 6 for line in table.splitlines() if line.startswith(('Baseline &', r'\textbf{Ours} &')))
    assert 'Mean Time' not in paper_table(result['summary'], 'trajectory')
    empty = {'Baseline': {}, 'Ours': {}}
    add_projection_timing(empty, [])
    assert empty['Baseline']['projection_latency_ms_mean'] == 0.
    assert empty['Ours']['projection_latency_ms_mean'] is None
    assert empty['Ours']['projection_latency_ms_p95'] is None


def test_cluster_bootstrap_uses_physical_frames():
    rows = make_records()
    # A second physical frame has no between-method difference. Different seeds
    # in the first frame must remain together, rather than becoming new clusters.
    more = [{**r, 'frame_index': 3} for r in rows]
    for r in rows:
        if r['method'] == 'Ours':
            r['joint_feasible'] = True
            r['joint_frame_feasible'] = True
    rows += [{**r, 'seed': 1} for r in rows] + more
    _, _, c = report(rows, 'step', 100, 5)
    assert c['joint_feasibility_pct']['bootstrap_95ci_pp'] == pytest.approx([0., 200/3])


def test_trajectory_summary_preserves_existing_metrics():
    rows = []
    for method in ('Baseline', 'Ours'):
        for ego in range(2):
            row = dict(method=method, seed=0, episode_index=1, frame_index=1, ego_index=ego,
                       joint_frame_feasible=False, joint_feasible=ego == 0, deviation_m=.2,
                       projection_applied=True, projection_control_delta_norm=0.,
                       projection_latency_ms=1., projection_iterations=1)
            for flag in FLAGS:
                row[flag] = False
            for name in ('geometric', 'stability', 'collision', 'external_collision', 'intertruck_collision',
                         'control', 'control_magnitude', 'control_rate'):
                row[name+'_violation'] = False
            for name in ('stability_excess', 'geometry_excess', 'control_magnitude_excess', 'control_rate_excess'):
                for agg in ('mean', 'max'):
                    row[name+'_'+agg+'_norm'] = 0.
            row.update(collision_penetration_mean_m=0., collision_penetration_max_m=0.)
            rows.append(row)
    from evaluation.ablation.jtp.statistics import summary
    before = _summary(rows[:2])
    after = summary(rows[:2], 'trajectory')
    assert all(after[k] == v for k, v in before.items())
    assert after['feasibility_pct'] == 50.
    assert after['joint_feasibility_pct'] == 0.
    assert 'tab:projection_ablation_trajectory' in paper_table(report(rows, 'trajectory', 0)[0])


@pytest.mark.parametrize('unit', [None, 'both', 'trajectory', 'step'])
def test_cli_statistics_selection(monkeypatch, unit):
    from evaluation.ablation.jtp import runner
    received = {}
    monkeypatch.setattr(runner, 'evaluate', lambda **kwargs: received.update(kwargs))
    args = ['--checkpoint', 'unused.pt']
    if unit is not None:
        args += ['--statistics-unit', unit]
    runner.main(args)
    assert received['statistics_unit'] == (unit or 'both')


def test_report_rejects_unpaired_and_duplicate_records():
    rows = make_records()
    with pytest.raises(ValueError, match='Unpaired'):
        report(rows[:-1], 'step')
    with pytest.raises(ValueError, match='Duplicate'):
        report(rows + rows[:1], 'step')
