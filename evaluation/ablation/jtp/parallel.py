"""Bounded, ordered frame execution with Windows/CUDA-safe spawn workers."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import multiprocessing
import torch


def cpu_payload(value):
    """Never transport CUDA tensors or autograd graphs between processes."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_payload(v) for v in value)
    if is_dataclass(value):
        return type(value)(**{f.name: cpu_payload(getattr(value, f.name)) for f in fields(value)})
    return value


_STATE = None


def initialize_worker(checkpoint, jtp_path, cfg, ev, device, threads):
    from highway_env.planner.diffusion.training.control import load_control_checkpoint
    from highway_env.planner.diffusion.projection import JointTrajectoryProjector
    global _STATE
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model, _ = load_control_checkpoint(checkpoint, device='cpu')
    model.to(device).eval()
    projector = JointTrajectoryProjector(model.vehicle_model, config=jtp_path)
    _STATE = (model, projector, cfg, ev, torch.device(device))


def evaluate_worker(job):
    from .runner import _evaluate_frame
    if _STATE is None:
        raise RuntimeError('Frame worker was not initialized')
    return cpu_payload(_evaluate_frame(job, *_STATE))


def ordered_bounded_map(executor, fn, jobs, max_pending):
    """Preserve input order and bound pending jobs and returned tensor storage."""
    iterator = iter(jobs)
    pending = deque()
    try:
        for _ in range(max_pending):
            job = next(iterator, None)
            if job is None:
                break
            pending.append((job, executor.submit(fn, job)))
        while pending:
            job, future = pending.popleft()
            result = future.result()
            yield job, result
            job = next(iterator, None)
            if job is not None:
                pending.append((job, executor.submit(fn, job)))
    finally:
        for _, future in pending:
            future.cancel()


@contextmanager
def frame_results(jobs, serial_fn, workers, initializer_args):
    if workers == 1:
        yield ((job, serial_fn(job)) for job in jobs)
        return
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=initializer_args) as executor:
        results = ordered_bounded_map(executor, evaluate_worker, jobs, 2*workers)
        try:
            yield results
        finally:
            results.close()
