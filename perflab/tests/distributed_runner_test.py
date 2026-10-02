# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for standalone distributed benchmark runner."""

from collections.abc import Iterator
import contextlib
import importlib.util
import os
import sys
import types
from typing import Any
from unittest import mock
from xml.etree import ElementTree

from absl.testing import absltest
from absl.testing import parameterized
import portpicker
import torch
import torch.multiprocessing as mp
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from perflab import common
from perflab import discovery as discovery_lib
from perflab import distributed_runner
from perflab import export as export_lib
from perflab import known_failures
from perflab import measure as measure_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import steps
from perflab import target as target_lib
from perflab.tests import fakes
from tests import seed_test_utils


def setUpModule():
  discovery_lib.import_submodules(steps)


@contextlib.contextmanager
def _single_rank_gloo_env(rank: int = 0) -> Iterator[None]:
  """Configures single-process Gloo rendezvous environment variables."""
  port = portpicker.pick_unused_port()
  with mock.patch.dict(
      os.environ,
      {
          "MASTER_ADDR": "127.0.0.1",
          "MASTER_PORT": str(port),
          "RANK": str(rank),
          "LOCAL_RANK": "0",
          "WORLD_SIZE": "1",
      },
  ):
    try:
      yield
    finally:
      if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _in_process_dist_run(nprocs: int, fn: Any, *args: Any) -> None:
  """In-process stand-in for `dist_run` that executes rank 0 over Gloo."""
  assert nprocs == 1
  with _single_rank_gloo_env(rank=0):
    fn(*args)


class _FakeDistRun:
  """Stateful fake for `distributed_utils.dist_run`.

  Records the spec name of every dispatched target, simulates a worker rank
  raising (as surfaced by `torch.multiprocessing.spawn`) for specs listed in
  `failing_specs`, and simulates a factory raising `UnsupportedBenchmark` for
  specs listed in `unsupported_specs`.
  """

  def __init__(
      self,
      failing_specs: set[str],
      unsupported_specs: frozenset[str] = frozenset(),
  ):
    self._failing_specs = failing_specs
    self._unsupported_specs = unsupported_specs
    self.ran = []

  def __call__(
      self,
      nproc_per_node,
      fn,
      spec_name,
      mode_value,
      platform_name,
      skip_reason_file,
  ):
    del nproc_per_node, fn, mode_value, platform_name
    self.ran.append(spec_name)
    if spec_name in self._unsupported_specs:
      # What rank 0 does when the factory raises UnsupportedBenchmark.
      with open(skip_reason_file, "w") as f:
        f.write(f"{spec_name} needs more chips")
    if spec_name in self._failing_specs:
      raise mp.ProcessRaisedException(
          f"RuntimeError: factory failed for {spec_name}",
          error_index=0,
          error_pid=1234,
      )


def _read_test_xml(path: str) -> dict[str, tuple[str, str]]:
  """Returns {test case name: (status, message)} from a JUnit XML report."""
  cases = {}
  for case in ElementTree.parse(path).getroot().iter("testcase"):
    status, message = "PASSED", ""
    for tag in ("failure", "skipped"):
      if (child := case.find(tag)) is not None:
        status, message = tag.upper(), child.get("message", "")
    cases[case.get("name")] = (status, message)
  return cases


class DistributedRunnerTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # Under blaze, XML_OUTPUT_FILE is this test's own report; main() must not
    # overwrite it.
    self.xml_path = os.path.join(self.create_tempdir().full_path, "test.xml")
    self.enter_context(
        mock.patch.dict(os.environ, {"XML_OUTPUT_FILE": self.xml_path})
    )
    self.exporter = fakes.RecordingExporter()
    self.enter_context(mock.patch.object(export_lib, "export", self.exporter))
    self.enter_context(mock.patch.dict(registry_lib.REGISTRY, {}, clear=True))

  def test_resolve_benchmark_targets_deterministic(self):
    registry_lib.REGISTRY["llama_8b_inference"] = fakes.tiny_spec(
        "llama_8b_inference"
    )

    # No filter returns all modes
    targets = distributed_runner.resolve_benchmark_targets(test_filter=None)
    self.assertLen(targets, len(common.RunMode))
    self.assertEqual(targets[0][0].name, "llama_8b_inference")

    # Exact target_name match
    targets = distributed_runner.resolve_benchmark_targets(
        test_filter=(
            "llama_8b_inference_eager_default"
            " llama_8b_inference_eager_optimized"
        )
    )
    self.assertLen(targets, 2)
    self.assertEqual(targets[0][1], common.RunMode.EAGER_DEFAULT)
    self.assertEqual(targets[1][1], common.RunMode.EAGER_OPTIMIZED)

    # Duplicate targets preserved in order
    targets = distributed_runner.resolve_benchmark_targets(
        test_filter=(
            "llama_8b_inference_eager_default llama_8b_inference_eager_default"
        )
    )
    self.assertLen(targets, 2)
    self.assertEqual(targets[0][1], common.RunMode.EAGER_DEFAULT)
    self.assertEqual(targets[1][1], common.RunMode.EAGER_DEFAULT)

    # pyformat: disable
    # Spec name without mode suffix throws ValueError
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=Runner target resolution validation.
      distributed_runner.resolve_benchmark_targets(
          test_filter="llama_8b_inference"
      )

    # Invalid target throws ValueError
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=Runner target resolution validation.
      distributed_runner.resolve_benchmark_targets(
          test_filter="invalid_target_xyz"
      )
    # pyformat: enable

  def test_resolve_benchmark_targets_wildcards(self):
    # Registered out of name order to check that targets are sorted by name
    for name in ("llama_8b_inference", "gemma_2b_training"):
      registry_lib.REGISTRY[name] = fakes.tiny_spec(name)

    # Targets are in token order, and a wildcard's matches in name order
    targets = distributed_runner.resolve_benchmark_targets(
        test_filter="llama_8b_inference_compiled *_eager_default"
    )
    self.assertEqual(
        [(spec.name, mode) for spec, mode in targets],
        [
            ("llama_8b_inference", common.RunMode.COMPILED),
            ("gemma_2b_training", common.RunMode.EAGER_DEFAULT),
            ("llama_8b_inference", common.RunMode.EAGER_DEFAULT),
        ],
    )

    # A wildcard can match every mode of a benchmark, in run mode order
    targets = distributed_runner.resolve_benchmark_targets(
        test_filter="gemma_2b_training_*"
    )
    self.assertEqual(
        [(spec.name, mode) for spec, mode in targets],
        [("gemma_2b_training", mode) for mode in common.RunMode],
    )

    # pyformat: disable
    # Wildcard pattern matching no target throws ValueError
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=Runner target resolution validation.
      distributed_runner.resolve_benchmark_targets(
          test_filter="*_no_such_mode"
      )
    # pyformat: enable

  def test_backend_by_device_kind(self):
    self.assertEqual(
        distributed_runner._BACKEND_BY_DEVICE_KIND,
        {
            target_lib.DeviceKind.CPU: "gloo",
            target_lib.DeviceKind.CUDA: "nccl",
            target_lib.DeviceKind.TPU: "tpu_dist",
        },
    )

  def test_run_single_target_task_end_to_end_exports_metrics(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")

    with _single_rank_gloo_env():
      distributed_runner._run_single_target_task(
          spec_name="tiny_linear",
          mode_value="eager_default",
          platform_name="cpu",
      )

    self.assertFalse(torch.distributed.is_initialized())
    self.assertLen(self.exporter.exported, 1)
    record = self.exporter.exported[0]
    self.assertEqual(record.spec_name, "tiny_linear")
    self.assertEqual(record.platform, target_lib.Platform.CPU)
    self.assertEqual(record.run_mode, common.RunMode.EAGER_DEFAULT)
    self.assertTrue(record.succeeded)
    self.assertGreater(record.metrics.e2e_wall_time_seconds, 0.0)
    self.assertGreater(record.metrics.first_step_time_seconds, 0.0)
    self.assertGreater(record.metrics.post_warmup_step_time_seconds, 0.0)

  def test_run_single_target_task_skips_unsupported_mode(self):
    registry_lib.REGISTRY["tiny_gpu"] = fakes.tiny_spec("tiny_gpu")

    distributed_runner._run_single_target_task(
        spec_name="tiny_gpu",
        mode_value="eager_optimized",
        platform_name="b200_8",
    )

    self.assertFalse(torch.distributed.is_initialized())
    self.assertEmpty(self.exporter.exported)

  def test_run_single_target_task_rejects_zero_metrics_on_rank_0(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    frozen_clock = fakes.FakeClock(start=10.0)
    orig_measure = measure_lib.measure

    with (
        _single_rank_gloo_env(),
        mock.patch.object(
            measure_lib,
            "measure",
            lambda *args, **kwargs: orig_measure(
                *args, clock=frozen_clock, **kwargs
            ),
        ),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Rank 0 zero-metric guard.
          AssertionError, "Invalid metrics measured"
      ):
        distributed_runner._run_single_target_task(
            spec_name="tiny_linear",
            mode_value="eager_default",
            platform_name="cpu",
        )

    self.assertFalse(torch.distributed.is_initialized())
    # The launcher in main() exports the failure, not the worker.
    self.assertEmpty(self.exporter.exported)

  def test_run_single_target_task_exception_propagates_and_cleans_up(self):
    def failing_factory(unused_ctx):
      raise RuntimeError("Model OOM")

    registry_lib.REGISTRY["oom_spec"] = fakes.tiny_spec(
        "oom_spec", factory=failing_factory
    )

    with _single_rank_gloo_env():
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Worker exception propagation and cleanup test.
          RuntimeError, "Model OOM"
      ):
        distributed_runner._run_single_target_task(
            spec_name="oom_spec",
            mode_value="eager_default",
            platform_name="cpu",
        )

    self.assertFalse(torch.distributed.is_initialized())
    # The launcher in main() exports the failure, not the worker.
    self.assertEmpty(self.exporter.exported)

  def test_run_single_target_task_reports_unsupported_benchmark(self):
    def unsupported_factory(unused_ctx):
      raise target_lib.UnsupportedBenchmark("needs more chips")

    registry_lib.REGISTRY["too_big"] = fakes.tiny_spec(
        "too_big", factory=unsupported_factory
    )
    skip_reason_file = os.path.join(self.create_tempdir().full_path, "skip.txt")

    with _single_rank_gloo_env():
      distributed_runner._run_single_target_task(
          spec_name="too_big",
          mode_value="eager_default",
          platform_name="cpu",
          skip_reason_file=skip_reason_file,
      )

    self.assertFalse(torch.distributed.is_initialized())
    self.assertEmpty(self.exporter.exported)
    with open(skip_reason_file) as f:
      self.assertEqual(f.read(), "needs more chips")

  def test_main_runs_filtered_target_and_clears_caches(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    custom_cache = os.path.join(
        self.create_tempdir().full_path, "inductor_cache"
    )
    os.makedirs(custom_cache)

    with (
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "cpu",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
                "TORCHINDUCTOR_CACHE_DIR": custom_cache,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=_in_process_dist_run,
        ),
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertFalse(os.path.exists(custom_cache))
    self.assertLen(self.exporter.exported, 1)
    record = self.exporter.exported[0]
    self.assertEqual(record.spec_name, "tiny_linear")
    self.assertEqual(record.platform, target_lib.Platform.CPU)
    self.assertEqual(record.run_mode, common.RunMode.EAGER_DEFAULT)
    self.assertTrue(record.succeeded)

  @mock.patch.object(mp, "spawn")
  def test_main_multihost_tpu_with_pinned_env_and_no_worker_hostnames(
      self, mock_mp_spawn
  ):
    registry_lib.REGISTRY["dummy"] = fakes.tiny_spec("dummy")
    pinned_env = {
        "BENCHMARK_PLATFORM": "v7_2x2x2",
        "TESTBRIDGE_TEST_ONLY": "dummy_eager_default",
        "JOB_COMPLETION_INDEX": "0",
        "NNODES": "2",
        "NODE_RANK": "0",
        "WORLD_SIZE": "16",
        "MASTER_ADDR": "host0",
        "MASTER_PORT": "29500",
        "TORCH_TPU_TOPOLOGY": "2,2,2,2",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": "host0:8070,host1:8070",
    }

    with (
        mock.patch.dict(os.environ, pinned_env, clear=True),
        mock.patch.object(
            distributed_runner.distributed_utils.env,
            "get_str_env_once",
            side_effect=lambda name, default_value=None: os.environ.get(
                name, default_value
            ),
        ),
    ):
      distributed_runner.main(["distributed_runner"])

    mock_mp_spawn.assert_called_once()
    self.assertEmpty(self.exporter.exported)

  def test_main_tpu_launch_env_singlehost_vs_multihost(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    dispatched: list[tuple[int, str, str, str]] = []

    def record_dispatch(
        nprocs, unused_fn, spec_name, mode_val, platform_val, unused_skip_file
    ):
      dispatched.append((nprocs, spec_name, mode_val, platform_val))

    # Single-host TPU (nnodes=1, nprocs_per_node=8) populates TPU launch env.
    with (
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "v7_2x2x1",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
                "TORCH_TPU_TOPOLOGY": "2x2x1",
            },
            clear=False,
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=record_dispatch,
        ),
    ):
      os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
      distributed_runner.main(["distributed_runner"])
      slicebuilder_addrs = os.environ.get(
          "TORCH_TPU_SLICEBUILDER_ADDRESSES", ""
      )
      self.assertLen(slicebuilder_addrs.split(","), 8)

    # Multi-host TPU (nnodes=2) does not override single-host TPU launch env.
    with (
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "v7_2x2x2",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
                "TORCH_TPU_TOPOLOGY": "2x2x2",
            },
            clear=False,
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=record_dispatch,
        ),
    ):
      os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
      distributed_runner.main(["distributed_runner"])
      self.assertNotIn("TORCH_TPU_SLICEBUILDER_ADDRESSES", os.environ)

    self.assertEqual(
        dispatched,
        [
            (8, "tiny_linear", "eager_default", "v7_2x2x1"),
            (8, "tiny_linear", "eager_default", "v7_2x2x2"),
        ],
    )

  def test_module_imports_without_torch_tpu_module(self):
    # Re-execute the module body as a GPU host with
    # TORCH_DEVICE_BACKEND_AUTOLOAD=0 would, where `torch.tpu` is not
    # registered. A module-level `torch.tpu` import would raise
    # ModuleNotFoundError here.
    spec = importlib.util.find_spec(distributed_runner.__name__)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules, {"torch.tpu": None, "torch.tpu.distributed": None}
    ):
      spec.loader.exec_module(module)

  def test_main_gpu_does_not_require_torch_tpu_module(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    dispatched: list[tuple[int, str, str, str]] = []

    def record_dispatch(
        nprocs, unused_fn, spec_name, mode_val, platform_val, unused_skip_file
    ):
      dispatched.append((nprocs, spec_name, mode_val, platform_val))

    # Simulate a GPU host with TORCH_DEVICE_BACKEND_AUTOLOAD=0, where
    # `torch.tpu` is not registered: any import of it raises ImportError.
    with (
        mock.patch.dict(
            sys.modules, {"torch.tpu": None, "torch.tpu.distributed": None}
        ),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=record_dispatch,
        ),
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertEqual(
        dispatched,
        [(8, "tiny_linear", "eager_default", "b200_8")],
    )

  def test_main_isolates_failed_target_and_exits_nonzero(self):
    fake_dist_run = _FakeDistRun(failing_specs={"broken"})
    registry = {
        "broken": types.SimpleNamespace(
            name="broken",
            skipped_run_modes=frozenset(),
            microbenchmark_name="batch_size_1",
        ),
        "healthy": types.SimpleNamespace(
            name="healthy", skipped_run_modes=frozenset()
        ),
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": (
                    "broken_eager_default healthy_eager_default"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertLogs(level="INFO") as logs,
        self.assertRaises(
            SystemExit
        ) as exit_cm,  # ASSERT_RAISES_OK=Runner must exit non-zero when any target fails.
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertNotEqual(exit_cm.exception.code, 0)
    # The healthy target still runs after the broken one fails.
    self.assertEqual(fake_dist_run.ran, ["broken", "healthy"])
    # The failed target is reported once as FAILED to the export path.
    self.assertEqual(
        [
            (d.spec_name, d.run_mode, d.succeeded, d.microbenchmark_name)
            for d in self.exporter.exported
        ],
        [("broken", common.RunMode.EAGER_DEFAULT, False, "batch_size_1")],
    )
    summary = "\n".join(logs.output)
    self.assertRegex(summary, r"broken_eager_default\s+FAILED")
    self.assertRegex(summary, r"healthy_eager_default\s+PASSED")
    self.assertIn("1 passed, 1 failed, 0 skipped", summary)

  def test_main_reports_unsupported_mode_as_skipped(self):
    fake_dist_run = _FakeDistRun(failing_specs=set())
    registry = {
        "healthy": types.SimpleNamespace(
            name="healthy", skipped_run_modes=frozenset()
        ),
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                # eager_optimized is TPU-only, so it is skipped on CUDA.
                "TESTBRIDGE_TEST_ONLY": (
                    "healthy_eager_optimized healthy_eager_default"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertLogs(level="INFO") as logs,
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertEqual(fake_dist_run.ran, ["healthy"])
    summary = "\n".join(logs.output)
    self.assertRegex(
        summary,
        r"healthy_eager_optimized\s+SKIPPED\s+Run mode eager_optimized is not"
        r" supported on cuda",
    )
    self.assertIn("1 passed, 0 failed, 1 skipped", summary)

  def test_main_reports_known_failure_as_skipped(self):
    fake_dist_run = _FakeDistRun(failing_specs=set())
    registry = {
        "known_bad": types.SimpleNamespace(
            name="known_bad", skipped_run_modes=frozenset()
        ),
        "healthy": types.SimpleNamespace(
            name="healthy", skipped_run_modes=frozenset()
        ),
    }
    failures = (
        known_failures.KnownFailure(
            benchmark="known_bad",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.CUDA,
            run_modes=frozenset({common.RunMode.COMPILED}),
            reason="simulated CUDA compiled failure",
        ),
    )

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.object(known_failures, "KNOWN_FAILURES", failures),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": (
                    "known_bad_compiled known_bad_eager_default"
                    " healthy_compiled"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertLogs(level="INFO") as logs,
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertEqual(fake_dist_run.ran, ["known_bad", "healthy"])
    summary = "\n".join(logs.output)
    self.assertRegex(
        summary,
        r"known_bad_compiled\s+SKIPPED\s+Benchmark known_bad is known to fail:"
        r" simulated CUDA compiled failure",
    )
    self.assertRegex(summary, r"known_bad_eager_default\s+PASSED")
    self.assertRegex(summary, r"healthy_compiled\s+PASSED")
    self.assertIn("2 passed, 0 failed, 1 skipped", summary)

  @parameterized.named_parameters(
      ("first_shard_node_0", "0", "0", True),
      ("second_shard", "1", "0", False),
      ("node_1", "0", "1", False),
  )
  def test_main_exports_known_failure_skips_once(
      self, shard_index, node_rank, exports_skips
  ):
    registry = {
        "known_bad": types.SimpleNamespace(
            name="known_bad", skipped_run_modes=frozenset()
        ),
        "skips_eager": types.SimpleNamespace(
            name="skips_eager", skipped_run_modes=frozenset({"eager_default"})
        ),
    }
    # Matches every run mode, including eager_optimized, which CUDA does not
    # run.
    failures = (
        known_failures.KnownFailure(
            benchmark="known_bad",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.CUDA,
            reason="simulated CUDA failure",
        ),
    )
    status_file = os.path.join(self.create_tempdir().full_path, "status")

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.object(known_failures, "KNOWN_FAILURES", failures),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": (
                    "known_bad_compiled known_bad_eager_optimized"
                    " skips_eager_eager_default"
                ),
                "TEST_TOTAL_SHARDS": "2",
                "TEST_SHARD_INDEX": shard_index,
                "TEST_SHARD_STATUS_FILE": status_file,
                "NODE_RANK": node_rank,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            _FakeDistRun(failing_specs=set()),
        ),
        mock.patch.object(
            export_lib, "export_skip", autospec=True
        ) as mock_export_skip,
    ):
      distributed_runner.main(["distributed_runner"])

    if exports_skips:
      mock_export_skip.assert_called_once_with(
          spec_name="known_bad",
          platform=target_lib.Platform.B200_8,
          framework=mode_lib.Framework.TORCH,
          run_mode=common.RunMode.COMPILED,
      )
    else:
      mock_export_skip.assert_not_called()

  def _run_main_sharded(
      self, total_shards: int, shard_index: int, status_file: str
  ) -> tuple[_FakeDistRun, str]:
    """Runs main() as one blaze test shard and returns (dist_run, summary)."""
    fake_dist_run = _FakeDistRun(failing_specs=set())
    registry = {
        name: types.SimpleNamespace(name=name, skipped_run_modes=frozenset())
        for name in ("a", "b", "c")
    }
    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                # eager_optimized is TPU-only, so it is skipped on CUDA and must
                # not count towards any shard.
                "TESTBRIDGE_TEST_ONLY": (
                    "a_eager_optimized a_compiled b_eager_optimized"
                    " b_compiled c_compiled"
                ),
                "TEST_TOTAL_SHARDS": str(total_shards),
                "TEST_SHARD_INDEX": str(shard_index),
                "TEST_SHARD_STATUS_FILE": status_file,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertLogs(level="INFO") as logs,
    ):
      distributed_runner.main(["distributed_runner"])
    return fake_dist_run, "\n".join(logs.output)

  @parameterized.named_parameters(
      ("shard_0", 0, ["a", "c"]),
      ("shard_1", 1, ["b"]),
  )
  def test_main_shards_runnable_targets_round_robin(
      self, shard_index, expected_ran
  ):
    status_file = os.path.join(self.create_tempdir().full_path, "status")

    fake_dist_run, summary = self._run_main_sharded(
        total_shards=2, shard_index=shard_index, status_file=status_file
    )

    self.assertEqual(fake_dist_run.ran, expected_ran)
    # Blaze treats a sharded test that never touches the status file as not
    # supporting sharding.
    self.assertTrue(os.path.exists(status_file))
    # Every shard reports the targets skipped for the platform.
    self.assertRegex(summary, r"a_eager_optimized\s+SKIPPED")
    self.assertRegex(summary, r"b_eager_optimized\s+SKIPPED")
    self.assertIn(f"{len(expected_ran)} passed, 0 failed, 2 skipped", summary)

  def test_main_shard_without_targets_passes(self):
    status_file = os.path.join(self.create_tempdir().full_path, "status")

    # 3 runnable targets across 4 shards leaves shard 3 empty.
    fake_dist_run, summary = self._run_main_sharded(
        total_shards=4, shard_index=3, status_file=status_file
    )

    self.assertEmpty(fake_dist_run.ran)
    self.assertTrue(os.path.exists(status_file))
    self.assertIn("0 passed, 0 failed, 2 skipped", summary)

  def test_main_writes_test_xml_per_target(self):
    fake_dist_run = _FakeDistRun(failing_specs={"broken"})
    registry = {
        name: types.SimpleNamespace(
            name=name, skipped_run_modes=frozenset(), microbenchmark_name=None
        )
        for name in ("broken", "healthy")
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                # eager_optimized is TPU-only, so it is skipped on CUDA.
                "TESTBRIDGE_TEST_ONLY": (
                    "broken_compiled healthy_compiled healthy_eager_optimized"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertRaises(
            SystemExit
        ),  # ASSERT_RAISES_OK=Runner must exit non-zero when any target fails.
    ):
      distributed_runner.main(["distributed_runner"])

    cases = _read_test_xml(self.xml_path)
    self.assertEqual(
        set(cases),
        {"broken_compiled", "healthy_compiled", "healthy_eager_optimized"},
    )
    self.assertEqual(cases["broken_compiled"][0], "FAILURE")
    self.assertIn("factory failed for broken", cases["broken_compiled"][1])
    self.assertEqual(cases["healthy_compiled"], ("PASSED", ""))
    self.assertEqual(
        cases["healthy_eager_optimized"],
        ("SKIPPED", "Run mode eager_optimized is not supported on cuda"),
    )
    suite = ElementTree.parse(self.xml_path).getroot().find("testsuite")
    self.assertEqual(suite.get("tests"), "3")
    self.assertEqual(suite.get("failures"), "1")
    self.assertEqual(suite.get("skipped"), "1")

  @parameterized.named_parameters(
      (
          "shard_0",
          0,
          {
              "a_compiled": "PASSED",
              "c_compiled": "PASSED",
              "a_eager_optimized": "SKIPPED",
              "b_eager_optimized": "SKIPPED",
          },
      ),
      ("shard_1", 1, {"b_compiled": "PASSED"}),
  )
  def test_main_test_xml_reports_skipped_targets_on_first_shard_only(
      self, shard_index, expected_cases
  ):
    status_file = os.path.join(self.create_tempdir().full_path, "status")

    self._run_main_sharded(
        total_shards=2, shard_index=shard_index, status_file=status_file
    )

    self.assertEqual(
        {
            name: status
            for name, (status, _) in _read_test_xml(self.xml_path).items()
        },
        expected_cases,
    )

  def test_shard_targets_returns_all_targets_without_sharding_env(self):
    targets = [
        (types.SimpleNamespace(name=name), common.RunMode.COMPILED)
        for name in ("a", "b", "c")
    ]
    self.enter_context(mock.patch.dict(os.environ))
    for k in (
        "TEST_TOTAL_SHARDS",
        "TEST_SHARD_INDEX",
        "TEST_SHARD_STATUS_FILE",
    ):
      os.environ.pop(k, None)

    self.assertEqual(distributed_runner._shard_targets(targets), targets)

  @parameterized.named_parameters(("negative", "-1"), ("too_large", "2"))
  def test_shard_env_rejects_out_of_range_index(self, shard_index):
    self.enter_context(
        mock.patch.dict(
            os.environ,
            {"TEST_TOTAL_SHARDS": "2", "TEST_SHARD_INDEX": shard_index},
        )
    )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Shard env validation.
        ValueError, "out of range"
    ):
      distributed_runner._shard_env()

  @parameterized.named_parameters(
      ("node_0", "0", True),
      ("node_1", "1", False),
  )
  def test_main_reports_factory_unsupported_benchmark_as_skipped(
      self, node_rank, exports_skip
  ):
    fake_dist_run = _FakeDistRun(
        failing_specs=set(), unsupported_specs=frozenset({"too_big"})
    )
    registry = {
        name: types.SimpleNamespace(name=name, skipped_run_modes=frozenset())
        for name in ("too_big", "healthy")
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": "too_big_compiled healthy_compiled",
                # Runtime skips happen on the shard that ran the target, so
                # even a non-first shard reports them.
                "TEST_TOTAL_SHARDS": "2",
                "TEST_SHARD_INDEX": "1",
                "NODE_RANK": node_rank,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        mock.patch.object(distributed_runner, "_shard_targets", lambda t: t),
        mock.patch.object(
            export_lib, "export_skip", autospec=True
        ) as mock_export_skip,
        self.assertLogs(level="INFO") as logs,
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertEqual(fake_dist_run.ran, ["too_big", "healthy"])
    if exports_skip:
      mock_export_skip.assert_called_once_with(
          spec_name="too_big",
          platform=target_lib.Platform.B200_8,
          framework=mode_lib.Framework.TORCH,
          run_mode=common.RunMode.COMPILED,
      )
    else:
      mock_export_skip.assert_not_called()
    summary = "\n".join(logs.output)
    self.assertRegex(
        summary, r"too_big_compiled\s+SKIPPED\s+too_big needs more chips"
    )
    self.assertIn("1 passed, 0 failed, 1 skipped", summary)
    self.assertEqual(
        _read_test_xml(self.xml_path),
        {
            "too_big_compiled": ("SKIPPED", "too_big needs more chips"),
            "healthy_compiled": ("PASSED", ""),
        },
    )

  def test_write_test_xml_strips_invalid_control_characters(self):
    distributed_runner._write_test_xml(
        self.xml_path,
        target_lib.Platform.B200_8,
        [
            distributed_runner._TargetResult(
                "broken_compiled",
                distributed_runner.TargetStatus.FAILED,
                "\x1b[31mRuntimeError\x1b[0m: bad\x00 value",
            )
        ],
    )

    self.assertEqual(
        _read_test_xml(self.xml_path),
        {"broken_compiled": ("FAILURE", "[31mRuntimeError[0m: bad value")},
    )

  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  def test_cleanup_distributed_worker_state_skip_barrier_on_exception(
      self,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
  ):
    del mock_compiler_reset, mock_is_initialized
    distributed_runner._cleanup_distributed_worker_state(
        rank=0, barrier_policy=distributed_runner.BarrierPolicy.SKIP
    )

    mock_barrier.assert_not_called()
    mock_destroy_group.assert_called_once()

  @parameterized.named_parameters(
      ("tpu", target_lib.Platform.V7_2X2X1, 1),
      ("cuda", target_lib.Platform.B200_8, 0),
      ("cpu", target_lib.Platform.CPU, 0),
  )
  def test_cleanup_distributed_worker_state_resets_torch_tpu_only_on_tpu(
      self, platform, expected_tpu_cleanups
  ):
    eager_state_resets = []
    tpu_cache_clears = []
    with (
        mock.patch.object(
            tt_testing,
            "reset_eager_state",
            lambda: eager_state_resets.append(True),
        ),
        mock.patch.object(
            torch.tpu, "_clear_cache", lambda: tpu_cache_clears.append(True)
        ),
    ):
      distributed_runner._cleanup_distributed_worker_state(
          rank=0,
          target=target_lib.make_target(platform),
          barrier_policy=distributed_runner.BarrierPolicy.SKIP,
      )

    self.assertLen(eager_state_resets, expected_tpu_cleanups)
    self.assertLen(tpu_cache_clears, expected_tpu_cleanups)

  def test_clear_persistent_compilation_caches(self):
    root = self.create_tempdir().full_path
    shm_cache = os.path.join(root, "torch_tpu_cache")
    inductor_cache = os.path.join(root, "torchinductor_user")
    compile_cache = os.path.join(root, "torch_compile_debug")
    custom_cache = os.path.join(root, "custom_inductor")
    unrelated_dir = os.path.join(root, "keep_me")
    for path in (
        shm_cache,
        inductor_cache,
        compile_cache,
        custom_cache,
        unrelated_dir,
    ):
      os.makedirs(path)

    with mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": custom_cache}):
      distributed_runner._clear_persistent_compilation_caches(
          shm_cache_dir=shm_cache,
          cache_patterns=(
              os.path.join(root, "torchinductor*"),
              os.path.join(root, "torch_compile*"),
          ),
      )

    self.assertFalse(os.path.exists(shm_cache))
    self.assertFalse(os.path.exists(inductor_cache))
    self.assertFalse(os.path.exists(compile_cache))
    self.assertFalse(os.path.exists(custom_cache))
    self.assertTrue(os.path.exists(unrelated_dir))


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
