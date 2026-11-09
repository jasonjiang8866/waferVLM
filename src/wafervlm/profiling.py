"""Opt-in synchronized phase timings; does not change the training recipe."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import resource
import time

from .core import write_json


class PhaseRecorder:
    def __init__(self, output, device):
        self.output, self.device = output, device
        self.events = []
        self.microstep = 0
        self.kernel_capture_done = False
        self.originals = []

    def sync(self):
        if self.device == 'cuda':
            import torch
            torch.cuda.synchronize()

    @contextmanager
    def scope(self, name):
        self.sync()
        started, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            self.sync()
            self.events.append({'phase': name, 'microstep': self.microstep,
                'wall_seconds': time.perf_counter() - started,
                'process_cpu_seconds': time.process_time() - cpu})

    def wrap(self, obj, name, phase):
        original = getattr(obj, name)
        self.originals.append((obj, name, original, name in vars(obj)))
        @wraps(original)
        def measured(*args, **kwargs):
            with self.scope(phase):
                return original(*args, **kwargs)
        setattr(obj, name, measured)

    def restore(self):
        for obj, name, original, was_instance_attribute in reversed(self.originals):
            if was_instance_attribute:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.originals.clear()

    def save(self):
        write_json(self.output / 'timings.json', {
            'schema': 1, 'device': self.device, 'synchronized': self.device == 'cuda',
            'events': self.events,
            'peak_host_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'notes': 'Inclusive nested timings; do not sum parent and child phases. '
                     'Process CPU time includes process threads, not system-wide utilization. '
                     'Warm-up is the first optimizer step. Kernel capture adds overhead.'})

    @contextmanager
    def capture_kernels(self, capture):
        if not capture or self.kernel_capture_done:
            yield
            return
        import torch
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    profile_memory=False, with_stack=False) as prof:
            yield
        events = prof.key_averages()
        rows = [{'name': x.key, 'count': x.count,
                 'self_cpu_ms': x.self_cpu_time_total / 1000,
                 'self_device_ms': x.self_device_time_total / 1000} for x in events]
        kernel_count = sum(str(e.device_type).endswith('CUDA') for e in prof.events())
        write_json(self.output / 'operators.json', {
            'captured_microstep': self.microstep, 'cuda_event_count': kernel_count,
            'top_cpu': sorted(rows, key=lambda r: r['self_cpu_ms'], reverse=True)[:30],
            'top_device': sorted(rows, key=lambda r: r['self_device_ms'], reverse=True)[:30],
            'notes': 'One microbatch after one optimizer-step warm-up; not end-to-end throughput.'})
        self.kernel_capture_done = True


def timed_trainer_class(recorder, accumulation):
    from transformers import Trainer

    class TimedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorder.wrap(self.accelerator, 'backward', 'backward')

        def _prepare_inputs(self, *args, **kwargs):
            with recorder.scope('host_to_device'):
                return super()._prepare_inputs(*args, **kwargs)

        def compute_loss(self, *args, **kwargs):
            with recorder.scope('forward_and_loss'):
                return super().compute_loss(*args, **kwargs)

        def training_step(self, *args, **kwargs):
            recorder.microstep += 1
            with recorder.capture_kernels(recorder.microstep == accumulation + 1):
                with recorder.scope('training_microbatch'):
                    return super().training_step(*args, **kwargs)

        def _save_checkpoint(self, *args, **kwargs):
            with recorder.scope('checkpoint_save'):
                return super()._save_checkpoint(*args, **kwargs)

    return TimedTrainer


def scope(recorder, name):
    return recorder.scope(name) if recorder is not None else nullcontext()
