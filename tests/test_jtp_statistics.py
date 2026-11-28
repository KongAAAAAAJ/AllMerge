import csv
import numpy as np
import pytest
from evaluation.ablation.jtp.distributions import KEYS, FLAGS
from evaluation.ablation.jtp.statistics import step_records, report, write_reports


def make_records():
    values = {k: np.zeros((2,4)) for k in KEYS}
    bad = {k: np.zeros((2,4),dtype=bool) for k in KEYS}
    bad['yaw'][0,0] = True
    bad['steering'][1,2] = True
    meta = [dict(seed=0,episode_index=1,frame_index=2,ego_index=i,payload_group='low') for i in range(2)]
    rows = []
    for method in ('Baseline','Ours'):
        rows.extend(step_records((values,bad),meta,method,np.zeros((2,4))))
    return rows


def test_six_constraints_include_first_step_and_joint_feasibility():
    rows = make_records()
    base = [r for r in rows if r['method']=='Baseline']
    assert len(base)==8
    assert sum(r['joint_feasible'] for r in base)==6
    assert sum(r['joint_frame_feasible'] for r in base)==4
    assert base[0]['yaw_violation']
    assert not any('road' in k or 'collision' in k or 'rate_violation' in k for k in base[0])
    result,delta,confidence = report(rows,'step',repeats=10)
    assert result['Baseline']['feasibility_pct']==75
    assert result['Baseline']['joint_feasibility_pct']==50
    assert delta['feasibility_pct']==0
    assert confidence['feasibility_pct']['bootstrap_95ci_pp']==[0.,0.]


def test_report_timing_table_and_saved_columns(tmp_path):
    rows = make_records()
    frames = [dict(seed=0,payload_group='low',optimization_time_ms_by_vehicle=[10.,20.])]
    result = write_reports(tmp_path,rows,'step',[0],5,0,frames)
    assert result['prediction_steps']=='1..H'
    assert result['summary']['Ours']['optimization_time_ms_mean']==15.
    table = (tmp_path/'step/projection_ablation_table_step.tex').read_text()
    assert 'Yaw rate' in table and 'Acceleration magnitude' in table
    assert all(word not in table for word in ('Geometry','Curvature','Collision','Acceleration rate','Steering rate'))
    with (tmp_path/'step/jtp_per_vehicle_step.csv').open(encoding='utf-8-sig') as stream:
        saved=list(csv.DictReader(stream))
    assert len(saved)==16
    assert all(row['prediction_step'] in ('1','2','3','4') for row in saved)


def test_report_rejects_unpaired_and_duplicate_records():
    rows=make_records()
    with pytest.raises(ValueError,match='Unpaired'):
        report(rows[:-1],'step',0)
    with pytest.raises(ValueError,match='Duplicate'):
        report(rows+[rows[0]],'step',0)


@pytest.mark.parametrize('unit',['both','trajectory','step'])
def test_cli_statistics_selection(monkeypatch,unit):
    from evaluation.ablation.jtp import runner
    args={}
    monkeypatch.setattr(runner,'evaluate',lambda **kwargs:args.update(kwargs))
    runner.main(['--checkpoint','unused.pt','--statistics-unit',unit])
    assert args['statistics_unit']==unit


def test_payload_summaries_use_independent_denominators(tmp_path):
    import json
    groups = ('low', 'medium', 'high')
    rows, frames = [], []
    for i, group in enumerate(groups):
        for frame in range(i+1):
            for row in make_records():
                row.update(payload_group=group, episode_index=i, frame_index=frame)
                if group == 'high':
                    row['yaw_violation'] = False
                rows.append(row)
            frames.append(dict(seed=0,payload_group=group,optimization_time_ms_by_vehicle=[10.*(i+1)]*2))
    result=write_reports(tmp_path,rows,'step',[0],5,0,frames)
    for i, group in enumerate(groups):
        methods=result['by_payload'][group]
        assert methods['Baseline']['num_samples']==8*(i+1)
        assert methods['Ours']['num_joint_samples']==4*(i+1)
        assert methods['Ours']['optimization_time_ms_mean']==10.*(i+1)
    assert result['by_payload']['high']['Baseline']['yaw_violation_pct']==0
    assert result['by_payload']['low']['Baseline']['yaw_violation_pct']==12.5
    table=(tmp_path/'step/projection_ablation_table_step.tex').read_text()
    assert '@{}lcccccc@{}' in table
    assert all(label in table for label in ('Low load','Medium load','High load'))
    assert table.count('Yaw rate satisfaction')==1
    pairs=json.loads((tmp_path/'step/jtp_paired_statistics_step_payload.json').read_text())
    assert set(pairs['by_payload'])==set(groups)
    with (tmp_path/'step/jtp_summary_step_payload.csv').open(encoding='utf-8-sig') as f:
        assert len(list(csv.DictReader(f)))==6
