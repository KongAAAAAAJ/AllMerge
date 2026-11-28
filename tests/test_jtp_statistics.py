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
    meta = [dict(seed=0,episode_index=1,frame_index=2,ego_index=i) for i in range(2)]
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
    frames = [dict(seed=0,optimization_time_ms_by_vehicle=[10.,20.])]
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
