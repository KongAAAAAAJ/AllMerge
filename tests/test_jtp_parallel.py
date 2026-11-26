from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import numpy as np
import pytest
import torch
from evaluation.ablation.jtp.parallel import cpu_payload, ordered_bounded_map, frame_results
from highway_env.planner.diffusion.projection import RoadGeometry


def square(value):
    return value*value


def fail(value):
    if value == 2:
        raise RuntimeError('synthetic frame failure')
    return value


def test_spawn_order_and_serial_equivalence():
    jobs = list(range(8))
    with frame_results(jobs, square, 1, ()) as results:
        expected = list(results)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as executor:
        actual = list(ordered_bounded_map(executor, square, jobs, 4))
    assert actual == expected


def test_failure_propagates_and_submission_is_bounded():
    consumed = []
    def jobs():
        for i in range(100):
            consumed.append(i)
            yield i
    with ThreadPoolExecutor(2) as executor:
        results = ordered_bounded_map(executor, fail, jobs(), 4)
        assert next(results) == (0, 0)
        assert len(consumed) == 4
        assert next(results) == (1, 1)
        with pytest.raises(RuntimeError, match='synthetic frame failure'):
            next(results)
        assert len(consumed) == 6


def test_cpu_payload_detaches_dataclasses_and_nested_results():
    x = torch.ones(2, requires_grad=True)
    road = RoadGeometry(torch.tensor([[0., 0.], [1., 0.]]), x, 2.)
    original = {'tensor': x*2, 'items': (road, [np.array([3])])}
    result = cpu_payload(original)
    assert result['tensor'].device.type == 'cpu'
    assert not result['tensor'].requires_grad
    assert not result['items'][0].left_width_m.requires_grad
    assert result['items'][1][0][0] == 3


@pytest.mark.parametrize('workers', [1, 2])
def test_worker_cli(monkeypatch, workers):
    from evaluation.ablation.jtp import runner
    received = {}
    monkeypatch.setattr(runner, 'evaluate', lambda **kwargs: received.update(kwargs))
    runner.main(['--checkpoint', 'unused.pt', '--workers', str(workers), '--worker-threads', '2'])
    assert received['workers'] == workers
    assert received['worker_threads'] == 2
