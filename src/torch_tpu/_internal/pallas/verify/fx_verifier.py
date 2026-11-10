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

"""Verification passes for `@jax_op` contracts, TPU lowering, FX graphs, and autograd.

This module implements the four PyTorch-boundary verification passes used by
`torch.tpu.pallas.verifier`:

1. `verify_op_contracts` (`Phase.CONTRACTS`): Validates eager storage aliasing
   on donated inputs, `nn.Parameter` buffer donation, mesh device uniqueness,
   `input_partition_specs` and `output_partition_specs` rank/axis/divisibility
   invariants, zero-output silent skip hazards, and custom `register_fake`
   shape/dtype/pytree agreement against `JaxCallable` runtime output buffers.
2. `verify_tpu_lowering` (`Phase.LOWERING`): Lowers a `@jax_op` to Mosaic TPU
   MLIR on the host via `jax.export` and `AbstractMesh`/`AbstractDevice`
   without requiring a physical TPU.
3. `verify_fx_graph` (`Phase.FX_GRAPH`): Traces arbitrary PyTorch callables or
   `nn.Module`s into an `fx.GraphModule` under `FakeTensorMode` and performs a
   forward storage-provenance dataflow analysis to detect `use_after_donate`,
   `aliased_donor_inputs`, and `donated_parameter` across views and tuples.
4. `verify_autograd_parity` (`Phase.AUTOGRAD`): Executes any custom backward
   registered via `op.register_autograd` on CPU TPU interpret mode and compares
   its gradients against `jax.vjp` of the forward JAX function (or central
   finite differences along Rademacher probes when `pallas_call`/`shard_map`
   lacks forward JVP rules).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import functools
import importlib
import inspect
import io
import math
import operator
import re
import sys
import textwrap
import threading
import tokenize
import types
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch
from torch._library import custom_ops
from torch._subclasses import fake_tensor
import torch.fx as fx
from torch.fx.experimental import proxy_tensor
from torch.utils import _python_dispatch
from torch.utils import _pytree as torch_pytree
from torch_tpu._internal.pallas import pallas as pallas_lib
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.utils import hardware

_CONTRACTS = findings.Phase.CONTRACTS
_FX_GRAPH = findings.Phase.FX_GRAPH
_AUTOGRAD = findings.Phase.AUTOGRAD
_LOWERING = findings.Phase.LOWERING

# Guards ref-counted installation of temporary process-level wrappers while
# `_THREAD_PATCH_STATE` keeps each active override strictly thread-local so
# concurrent calls on other threads pass straight through unpatched.
_PATCH_LOCK = threading.Lock()
_THREAD_PATCH_STATE = threading.local()
_PALLAS_CALL_STUB_REFCOUNT = 0
_ORIG_PALLAS_CALL: Any = None
_JC_EXPORTED_PATCHES: dict[int, tuple[pallas_lib.JaxCallable, Any, int]] = {}
_BACKWARD_BOUNDARY_OP: Any = None


def _get_backward_boundary_op() -> Any:
  """Lazily registers a no-op custom op marking the fwd->bwd FX boundary."""
  global _BACKWARD_BOUNDARY_OP
  if _BACKWARD_BOUNDARY_OP is None:
    with _PATCH_LOCK:
      if _BACKWARD_BOUNDARY_OP is None:
        op_def = torch.library.custom_op(
            "pallas_verify::_backward_boundary",
            lambda x: x.clone(),
            mutates_args=(),
            schema="(Tensor x) -> Tensor",
        )
        op_def.register_fake(lambda x: x.clone())
        _BACKWARD_BOUNDARY_OP = (
            torch.ops.pallas_verify._backward_boundary.default
        )
  return _BACKWARD_BOUNDARY_OP


# Longest lowering error message kept in a finding; Mosaic errors can embed
# whole MLIR modules.
_MAX_LOWERING_MESSAGE_CHARS = 1200

# Default TPU hardware generation that ops without a mesh are lowered for.
DEFAULT_TPU_VERSION = findings.DEFAULT_TPU_VERSION

# PyTorch and JAX 8-bit floating-point dtypes. NumPy has no native float8
# scalar types, so converting between torch.Tensor and jax.Array reinterprets
# raw uint8 bytes as ml_dtypes float8 types (see `tensor_to_numpy`).
FLOAT8_TORCH_DTYPES: frozenset[torch.dtype] = frozenset({
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
})
FLOAT8_JAX_DTYPES: frozenset[Any] = frozenset(
    pallas_lib.TORCH_TO_JAX_DTYPE_MAP[dt] for dt in FLOAT8_TORCH_DTYPES
)
INT_OR_BOOL_TORCH_DTYPES: frozenset[torch.dtype] = frozenset({
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
    torch.int32,
    torch.uint32,
    torch.int64,
    torch.uint64,
})


def is_fake_tensor(val: Any) -> bool:
  """Returns whether `val` is a PyTorch `FakeTensor`."""
  return isinstance(val, fake_tensor.FakeTensor)


MeshLike = jax.sharding.Mesh | jax.sharding.AbstractMesh


def _truncate(text: str, limit: int) -> str:
  """Trims whitespace and truncates `text` to at most `limit` characters."""
  text = text.strip()
  return text if len(text) <= limit else text[:limit] + " ..."


def _is_missing_tpu_device_kind_error(exc: BaseException) -> bool:
  """Returns whether `exc` is JAX/Pallas's missing TPU generation error."""
  return "Unsupported TPU device kind" in str(exc)


# ==============================================================================
# 1. Custom Op & Callable Introspection
# ==============================================================================


def resolve_custom_op_def(op_or_fn: Any) -> custom_ops.CustomOpDef | None:
  """Resolves a PyTorch `CustomOpDef` from an op handle, overload, or qualname.

  `@jax_op` returns a `CustomOpDef`, whereas FX graph `call_function` nodes
  record the underlying `OpOverload` (`torch.ops.<ns>.<name>.default`). This
  helper normalizes any of those representations back to the `CustomOpDef`.

  Args:
    op_or_fn: A `CustomOpDef`, `OpOverload`, `OpOverloadPacket`, or op name.

  Returns:
    The registered `CustomOpDef`, or `None` if `op_or_fn` is not a custom op.
  """
  if isinstance(op_or_fn, custom_ops.CustomOpDef):
    return op_or_fn
  if isinstance(op_or_fn, str):
    return custom_ops.OPDEFS.get(op_or_fn)
  schema = getattr(op_or_fn, "_schema", None)
  if hasattr(schema, "name"):
    qualname = str(schema.name).split(".", 1)[0]
    return custom_ops.OPDEFS.get(qualname)
  qualname = getattr(op_or_fn, "_qualified_op_name", None)
  if isinstance(qualname, str):
    return custom_ops.OPDEFS.get(qualname)
  return None


def extract_jax_callable(op_or_fn: Any) -> pallas_lib.JaxCallable | None:
  """Extracts the underlying TorchTPU `JaxCallable` if `op_or_fn` wraps one."""
  if isinstance(op_or_fn, pallas_lib.JaxCallable):
    return op_or_fn
  op_def = resolve_custom_op_def(op_or_fn)
  if op_def is not None and isinstance(op_def._init_fn, pallas_lib.JaxCallable):
    return op_def._init_fn
  return None


def get_op_display_name(op_or_fn: Any) -> str:
  """Returns a stable human-readable identifier for an op or function."""
  op_def = resolve_custom_op_def(op_or_fn)
  if op_def is not None:
    return op_def._qualname
  jax_callable = extract_jax_callable(op_or_fn)
  if jax_callable is not None:
    return jax_callable.name
  name = getattr(op_or_fn, "__name__", None)
  if isinstance(name, str):
    return name
  if isinstance(op_or_fn, functools.partial):
    return get_op_display_name(op_or_fn.func)
  return type(op_or_fn).__name__


def is_static_positional_arg(
    jax_callable: pallas_lib.JaxCallable, idx: int
) -> bool:
  """Returns whether positional argument `idx` is treated as static by `jax_callable`."""
  # pylint: disable=protected-access
  param = (
      jax_callable._orig_params[idx]
      if idx < len(jax_callable._orig_params)
      else None
  )
  return idx in jax_callable.static_argnums or (
      param is not None
      and (
          param.name in jax_callable._static_param_names
          or not jax_callable._param_is_tensor.get(param.name, False)
      )
  )
  # pylint: enable=protected-access


def _is_static_keyword_arg(
    jax_callable: pallas_lib.JaxCallable, name: str
) -> bool:
  """Returns whether keyword argument `name` is treated as static by `jax_callable`."""
  # pylint: disable=protected-access
  if name in jax_callable._static_param_names:
    return True
  param = jax_callable._orig_params_by_name.get(name)
  if param is not None and not jax_callable._param_is_tensor.get(name, False):
    return True
  param_idx = jax_callable._param_name_to_idx.get(name, -1)
  return param_idx >= 0 and is_static_positional_arg(jax_callable, param_idx)
  # pylint: enable=protected-access


def _get_donated_arg_indices(
    jax_callable: pallas_lib.JaxCallable,
    num_pos_args: int,
    *,
    include_io_aliases: bool = True,
) -> set[int]:
  """Returns normalized 0-based argument indices donated by `jax_callable`."""
  # pylint: disable=protected-access
  raw_indices = set(jax_callable._donate_argnums_set)
  io_alias_keys = set(jax_callable.input_output_aliases.keys())
  if include_io_aliases:
    raw_indices |= io_alias_keys
  bound_arity = max(num_pos_args, len(jax_callable._orig_params))
  # pylint: enable=protected-access
  normalized: set[int] = set()
  for idx in raw_indices:
    if idx < 0 and bound_arity > 0:
      normalized.add(idx + bound_arity)
    else:
      normalized.add(idx)
  if not include_io_aliases and io_alias_keys:
    for idx in io_alias_keys:
      norm_idx = idx + bound_arity if idx < 0 and bound_arity > 0 else idx
      normalized.discard(norm_idx)
  return normalized


# ==============================================================================
# 2. Target TPU Generation Inference & Abstract TPU Meshes
# ==============================================================================

# All TPU generations supported by TorchTPU (`hardware.TpuSpec`), in
# `TpuVersion` declaration order for deterministic candidate evaluation.
_CANDIDATE_TPU_VERSIONS: tuple[findings.TpuVersion, ...] = tuple(
    findings.TpuVersion
)

# TPU generations whose `TpuSpec` includes SparseCore hardware.
_SPARSE_CORE_TPU_VERSIONS: frozenset[findings.TpuVersion] = frozenset(
    version
    for version in findings.TpuVersion
    if hardware.TpuSpec.for_version(version).sparse_core_spec is not None
)

_DOC_TPU_PATTERNS: tuple[tuple[re.Pattern[str], findings.TpuVersion], ...] = (
    (re.compile(r"\btpu\s*v?8i\b", re.IGNORECASE), findings.TpuVersion.V8I),
    (re.compile(r"\btpu\s*v?8t\b", re.IGNORECASE), findings.TpuVersion.V8T),
    (re.compile(r"\btpu\s*v?7x?\b", re.IGNORECASE), findings.TpuVersion.V7),
    (
        re.compile(r"\b(?:tpu\s*v?6(?:e|\s+lite)?|trillium)\b", re.IGNORECASE),
        findings.TpuVersion.V6E,
    ),
    (re.compile(r"\btpu\s*v?5p\b", re.IGNORECASE), findings.TpuVersion.V5P),
    (
        re.compile(r"\btpu\s*v?5(?:e|\s+lite)\b", re.IGNORECASE),
        findings.TpuVersion.V5E,
    ),
)

_DOC_NEGATION_RE = re.compile(
    r"\b(?:not|no|except|excluding|unsupported|without|todo)\b",
    re.IGNORECASE,
)

# Standard library and framework module prefixes skipped when scanning user
# kernel modules for docstrings, comments, or `supported_on` predicates.
_SKIP_MODULE_PREFIXES = (
    "torch.",
    "torch_tpu.",
    "jax.",
    "jaxlib.",
    "numpy",
    "builtins",
    "functools",
    "typing",
    "collections",
    "abc",
)

# Wrapper attributes commonly used by decorators, JAX primitives, and custom ops
# to reference an underlying callable.
_CALLABLE_WRAPPER_ATTRS = (
    "__wrapped__",
    "_fun",
    "func",
    "f",
    "_fn",
    "fn",
    "__self__",
    "__func__",
    "op_impl_jax",
    "jit_fn",
)


def abstract_tpu_device(tpu_version: findings.TpuVersion | str) -> Any:
  """Returns an `AbstractDevice` targeting TPU `tpu_version`."""
  jax_kind = findings.jax_tpu_device_kind(tpu_version)
  parsed = (
      findings.parse_tpu_version(tpu_version)
      if isinstance(tpu_version, str)
      else tpu_version
  )
  # TPU v5p exposes two TensorCores per logical device in Megacore mode.
  num_cores = 2 if parsed is findings.TpuVersion.V5P else 1
  device_cls = jax.sharding.AbstractDevice
  if "platform" in inspect.signature(device_cls).parameters:
    return device_cls(jax_kind, num_cores, "tpu")  # type: ignore[call-arg]
  # JAX releases up to 0.9 take no platform argument.
  return device_cls(jax_kind, num_cores)  # pyrefly: ignore[missing-argument]


def abstract_tpu_mesh(tpu_version: findings.TpuVersion | str) -> Any:
  """Returns a one-device abstract mesh that targets TPU `tpu_version`."""
  return jax.sharding.AbstractMesh(
      (1,),
      ("x",),
      axis_types=(jax.sharding.AxisType.Auto,),
      abstract_device=abstract_tpu_device(tpu_version),
  )


def _strip_python_comments(src: str) -> str:
  """Strips `#` comments from Python source while preserving punctuation."""
  dedented = textwrap.dedent(src)
  lines = dedented.splitlines(keepends=True)
  comment_spans_by_row: dict[int, list[tuple[int, int]]] = {}
  try:
    for tok in tokenize.generate_tokens(io.StringIO(dedented).readline):
      if tok.type == tokenize.COMMENT:
        srow, scol = tok.start
        _, ecol = tok.end
        comment_spans_by_row.setdefault(srow - 1, []).append((scol, ecol))
  except (tokenize.TokenError, IndentationError):
    if not comment_spans_by_row:
      return "".join(
          line.split("#", 1)[0]
          + ("\n" if line.endswith("\n") and "#" in line else "")
          for line in lines
      )
  except SyntaxError:
    return "".join(
        line.split("#", 1)[0]
        + ("\n" if line.endswith("\n") and "#" in line else "")
        for line in lines
    )
  if not comment_spans_by_row:
    return dedented
  out_lines = list(lines)
  for row_idx, spans in comment_spans_by_row.items():
    if 0 <= row_idx < len(out_lines):
      line = out_lines[row_idx]
      for scol, ecol in reversed(spans):
        line = line[:scol] + line[ecol:]
      out_lines[row_idx] = line
  return "".join(out_lines)


def _extract_doc_tpu_versions(text: str) -> list[findings.TpuVersion]:
  """Extracts non-negated `TpuVersion` targets in docstring appearance order."""
  matches: list[tuple[int, findings.TpuVersion]] = []
  for pattern, canonical in _DOC_TPU_PATTERNS:
    for match in pattern.finditer(text):
      line_start = text.rfind("\n", 0, match.start()) + 1
      clause_prefix = re.split(
          r"[.;:]", text[max(line_start, match.start() - 48) : match.start()]
      )[-1]
      clause_tail = re.split(r"[.;:\n]", text[match.end() : match.end() + 32])[
          0
      ]
      if _DOC_NEGATION_RE.search(clause_prefix) or re.search(
          r"\btodo\b", clause_tail, re.IGNORECASE
      ):
        continue
      matches.append((match.start(), canonical))
  matches.sort(key=lambda item: item[0])
  found: list[findings.TpuVersion] = []
  for _, canonical in matches:
    if canonical not in found:
      found.append(canonical)
  return found


def _iter_callable_children(obj: Any) -> Iterator[Any]:
  """Yields direct child objects referenced by wrappers, partials, or closures."""
  jax_callable = extract_jax_callable(obj)
  if jax_callable is not None:
    if jax_callable is not obj:
      yield jax_callable
    yield jax_callable.jit_fn
  op_def = resolve_custom_op_def(obj)
  if op_def is not None:
    if op_def is not obj:
      yield op_def
    abstract_fn = getattr(op_def, "_abstract_fn", None)
    if abstract_fn is not None:
      yield abstract_fn

  for attr in _CALLABLE_WRAPPER_ATTRS:
    try:
      child = getattr(obj, attr, None)
    except Exception:  # pylint: disable=broad-exception-caught
      child = None
    if child is not None:
      yield child

  if isinstance(obj, torch.nn.Module):
    forward = getattr(obj, "forward", None)
    if forward is not None:
      yield forward
    yield from obj.children()
    for val in vars(obj).values():
      if extract_jax_callable(val) is not None or callable(val):
        yield val
  elif isinstance(obj, functools.partial):
    yield from obj.args
    yield from (obj.keywords or {}).values()
  elif isinstance(obj, (tuple, list, set, frozenset)):
    yield from obj
  elif isinstance(obj, Mapping):
    yield from obj.values()
  elif isinstance(obj, types.FunctionType):
    for cell in obj.__closure__ or ():
      try:
        yield cell.cell_contents
      except ValueError:
        continue
    for d in obj.__defaults__ or ():
      yield d
    for d in (obj.__kwdefaults__ or {}).values():
      yield d


def _iter_code_names(code: types.CodeType) -> Iterator[str]:
  """Yields global names referenced by `code` and nested code constants."""
  yield from code.co_names
  for const in code.co_consts:
    if isinstance(const, types.CodeType):
      yield from _iter_code_names(const)


def _collect_callable_objects(
    op_or_fn: Any, *, max_objects: int = 96
) -> list[Any]:
  """Collects reachable functions and callable wrappers starting from `op_or_fn`."""
  queue: list[Any] = [op_or_fn]

  leaf_types = (
      int,
      float,
      str,
      bytes,
      bool,
      torch.Tensor,
      jax.Array,
      np.ndarray,
  )
  visited_ids: set[int] = set()
  visited_objs: list[Any] = []

  while queue and len(visited_objs) < max_objects:
    obj = queue.pop(0)
    if obj is None or isinstance(obj, leaf_types):
      continue
    obj_id = id(obj)
    if obj_id in visited_ids:
      continue
    visited_ids.add(obj_id)
    visited_objs.append(obj)

    queue.extend(_iter_callable_children(obj))

    # Also follow same-module helper functions and `@jax_op` globals in `co_names`.
    if isinstance(obj, types.FunctionType):
      mod_name = getattr(obj, "__module__", "") or ""
      if not mod_name.startswith(_SKIP_MODULE_PREFIXES):
        gdict = getattr(obj, "__globals__", None) or {}
        for name in _iter_code_names(obj.__code__):
          gval = gdict.get(name)
          if extract_jax_callable(gval) is not None or (
              isinstance(gval, (types.FunctionType, type))
              and getattr(gval, "__module__", None) == mod_name
          ):
            queue.append(gval)

  return visited_objs


def _collect_user_modules(
    visited_objs: Sequence[Any],
) -> list[types.ModuleType]:
  """Returns user-defined modules (and sibling Pallas TPU modules) for `visited_objs`."""
  modules: list[types.ModuleType] = []
  seen_mod_names: set[str] = set()
  for obj in visited_objs:
    mod_name = getattr(obj, "__module__", None)
    if not isinstance(mod_name, str) or not mod_name:
      mod_name = getattr(type(obj), "__module__", None)
    if (
        not isinstance(mod_name, str)
        or not mod_name
        or mod_name.startswith(_SKIP_MODULE_PREFIXES)
    ):
      continue
    # Libraries such as Tokamax split the raw Pallas kernel (`*_kernel.py`) and
    # the `PallasMosaicTpu*` class defining `supported_on` (`pallas_mosaic_tpu.py`)
    # into sibling modules within the same subpackage.
    sibling_mod = (
        f"{mod_name.rsplit('.', 1)[0]}.pallas_mosaic_tpu"
        if "." in mod_name
        and mod_name.endswith(("_kernel", "torch_pallas_mosaic_tpu"))
        else None
    )
    for candidate_mod in (mod_name, sibling_mod):
      if not candidate_mod or candidate_mod in seen_mod_names:
        continue
      seen_mod_names.add(candidate_mod)
      mod = sys.modules.get(candidate_mod)
      if mod is None and candidate_mod.endswith(".pallas_mosaic_tpu"):
        try:
          mod = importlib.import_module(candidate_mod)
        except Exception:  # pylint: disable=broad-exception-caught
          mod = None
      if isinstance(mod, types.ModuleType):
        modules.append(mod)
  return modules


def _collect_kernel_metadata(
    visited_objs: Sequence[Any],
    modules: Sequence[types.ModuleType],
) -> tuple[list[Callable[[Any], Any]], list[str], list[str]]:
  """Extracts `(supported_on_fns, doc_texts, sources)` from kernel objects."""
  supported_on_fns: list[Callable[[Any], Any]] = []
  doc_texts: list[str] = []
  source_texts: list[str] = []

  for obj in visited_objs:
    if not isinstance(obj, type):
      sup = getattr(obj, "supported_on", None)
      if callable(sup):
        supported_on_fns.append(sup)
    mod_name = getattr(obj, "__module__", "") or ""
    if isinstance(obj, (types.FunctionType, type)) and not mod_name.startswith(
        _SKIP_MODULE_PREFIXES
    ):
      doc = inspect.getdoc(obj)
      if doc:
        doc_texts.append(doc)
      try:
        src = inspect.getsource(obj)
        source_texts.append(src)
      except (OSError, TypeError):
        pass

  for mod in modules:
    mod_doc = getattr(mod, "__doc__", None)
    if isinstance(mod_doc, str) and mod_doc:
      doc_texts.append(mod_doc)
    for fn_attr in ("_supported_on", "supported_on"):
      sup_fn = getattr(mod, fn_attr, None)
      if callable(sup_fn) and not isinstance(sup_fn, type):
        supported_on_fns.append(sup_fn)
    for val in vars(mod).values():
      if (
          isinstance(val, type)
          and getattr(val, "__module__", None) == mod.__name__
          and "supported_on" in val.__dict__
      ):
        raw_sup = val.__dict__["supported_on"]
        if isinstance(raw_sup, staticmethod):
          supported_on_fns.append(raw_sup.__func__)
        elif isinstance(raw_sup, classmethod):
          supported_on_fns.append(
              lambda dev, f=raw_sup.__func__, cls=val: f(cls, dev)
          )
        elif callable(raw_sup):
          supported_on_fns.append(lambda dev, f=raw_sup: f(None, dev))

  return supported_on_fns, doc_texts, source_texts


def _narrow_tpu_versions_by_predicates(
    candidates: Sequence[findings.TpuVersion],
    supported_on_fns: Sequence[Callable[[Any], Any]],
) -> tuple[list[findings.TpuVersion], bool]:
  """Filters `candidates` using the first evaluable `supported_on(device)` predicate."""
  allowed = list(candidates)
  for sup_fn in supported_on_fns:
    matched: list[findings.TpuVersion] = []
    evaluated = False
    for version in allowed:
      try:
        with override_abstract_mesh(abstract_tpu_mesh(version)):
          ok = bool(sup_fn(abstract_tpu_device(version)))
        evaluated = True
        if ok:
          matched.append(version)
      except Exception:  # pylint: disable=broad-exception-caught
        continue
    if evaluated and matched:
      return matched, True
  return allowed, False


def infer_explicit_tpu_version(
    op_or_fn: Any,
) -> findings.TpuVersion | None:
  """Returns the target `TpuVersion` inferred from `op_or_fn`, or `None`."""
  visited_objs = _collect_callable_objects(op_or_fn)
  for obj in visited_objs:
    jax_callable = extract_jax_callable(obj)
    if (
        jax_callable is not None
        and jax_callable.mesh is not None
        and jax_callable.mesh.size > 0
    ):
      mesh = jax_callable.mesh
      dev = (
          mesh.abstract_device
          if isinstance(mesh, jax.sharding.AbstractMesh)
          else mesh.devices.flat[0]
      )
      dev_kind = getattr(dev, "device_kind", None)
      if isinstance(dev_kind, str) and dev_kind:
        parsed = findings.parse_tpu_version(dev_kind)
        if parsed is not None:
          return parsed

  modules = _collect_user_modules(visited_objs)
  supported_on_fns, doc_texts, source_texts = _collect_kernel_metadata(
      visited_objs, modules
  )

  # 1. Narrow candidate TPU generations via `supported_on(device)` if present.
  allowed, narrowed = _narrow_tpu_versions_by_predicates(
      _CANDIDATE_TPU_VERSIONS, supported_on_fns
  )

  # 2. Restrict to TPU generations with SparseCore hardware if the kernel source
  # uses SparseCore APIs.
  if _sources_use_sparse_core(source_texts):
    sc_allowed = [v for v in allowed if v in _SPARSE_CORE_TPU_VERSIONS]
    if sc_allowed:
      allowed = sc_allowed
      narrowed = True

  # 3. Prefer any TPU generation explicitly named in docstrings.
  doc_mentions = _extract_doc_tpu_versions("\n".join(doc_texts))
  for doc_version in doc_mentions:
    if doc_version in allowed:
      return doc_version
  if doc_mentions and not narrowed:
    return doc_mentions[0]

  if narrowed and allowed:
    return (
        DEFAULT_TPU_VERSION if DEFAULT_TPU_VERSION in allowed else allowed[-1]
    )
  return None


def _sources_use_sparse_core(source_texts: Sequence[str]) -> bool:
  stripped_sources = [_strip_python_comments(src) for src in source_texts]
  return any(
      ".sparse_core" in src or "pallas.tpu_sc" in src or "import tpu_sc" in src
      for src in stripped_sources
  )


def _function_uses_sparse_core(op_or_fn: Any) -> bool:
  """Returns whether `op_or_fn` or its helpers invoke SparseCore APIs."""
  visited_objs = _collect_callable_objects(op_or_fn)
  modules = _collect_user_modules(visited_objs)
  _, _, source_texts = _collect_kernel_metadata(visited_objs, modules)
  return _sources_use_sparse_core(source_texts)


def infer_tpu_version(
    op_or_fn: Any,
    default_version: findings.TpuVersion | None = None,
) -> findings.TpuVersion:
  """Infers the target `TpuVersion` from `op_or_fn` or returns the default.

  When `default_version` is explicitly provided, it is returned directly.
  Otherwise, `op_or_fn` (along with its unwrapped callables, bound op instances,
  and defining module) is inspected for:

  1. A bound mesh whose devices report a TPU `device_kind`.
  2. A `supported_on(device)` predicate method or helper function.
  3. SparseCore (`pltpu.get_tpu_info().sparse_core` or `pallas.tpu_sc`) usage,
     which restricts candidates to TPU generations with a `sparse_core_spec`
     (excluding `TpuVersion.V5E`).
  4. Explicit target TPU mentions in kernel/op docstrings (such as `"TPU v8i"`,
     `"TPU v8t"`, `"TPU v7"`, `"TPU7x"`, `"TPU v6e"`, `"Trillium"`, `"TPU v5p"`,
     or `"TPU v5e"`).

  Args:
    op_or_fn: The `@jax_op`, JAX function, or PyTorch callable being verified.
    default_version: Explicit `TpuVersion` override, or `None` to infer.

  Returns:
    The target `TpuVersion`, falling back to `DEFAULT_TPU_VERSION`
    (`TpuVersion.V7`).
  """
  if default_version is not None:
    return default_version
  return infer_explicit_tpu_version(op_or_fn) or DEFAULT_TPU_VERSION


# ==============================================================================
# 3. Mesh Discovery, Context Managers, and CPU Mesh Rebinding
# ==============================================================================


def is_mock_tpu_mesh(mesh: MeshLike | None) -> bool:
  """Returns True if `mesh` uses non-CPU (TPU, abstract, or mock) devices.

  CPU interpret mode requires a concrete `jax.sharding.Mesh` backed by virtual
  CPU `jax.Device` instances. Any `AbstractMesh`, mock `_PallasDevice` mesh, or
  physical TPU mesh returns `True` so callers know to rebind it to a CPU mesh.

  Args:
    mesh: A JAX `Mesh`, `AbstractMesh`, or `None`.

  Returns:
    `True` if `mesh` is non-empty and not backed by concrete CPU `jax.Device`s.
  """
  if mesh is None or mesh.size == 0:
    return False
  if isinstance(mesh, jax.sharding.AbstractMesh):
    return True
  first_dev = mesh.devices.flat[0]
  return (
      not isinstance(first_dev, jax.Device)
      or getattr(first_dev, "platform", None) != "cpu"
  )


def make_cpu_mesh_for(
    mesh: MeshLike, *, preserve_axis_types: bool = True
) -> jax.sharding.Mesh | None:
  """Builds a CPU `jax.sharding.Mesh` matching `mesh` if enough CPU devices exist.

  Args:
    mesh: Source concrete or abstract JAX mesh whose topology should be matched.
    preserve_axis_types: Whether to preserve `mesh.axis_types` on the CPU mesh.

  Returns:
    A CPU `jax.sharding.Mesh` with the same shape and axis names, or `None` if
    fewer virtual CPU devices are available than `mesh.size`.
  """
  cpu_devs = jax.devices("cpu")
  if len(cpu_devs) < mesh.size:
    return None
  mesh_shape = tuple(int(mesh.shape[ax]) for ax in mesh.axis_names)
  reshaped = np.array(cpu_devs[: mesh.size]).reshape(mesh_shape)
  axis_types = (
      getattr(mesh, "axis_types", None) if preserve_axis_types else None
  )
  return jax.sharding.Mesh(reshaped, mesh.axis_names, axis_types=axis_types)


@contextlib.contextmanager
def override_abstract_mesh(mesh: jax.sharding.AbstractMesh):
  """Binds `mesh` via `use_abstract_mesh`, clearing any conflicting ambient mesh.

  JAX raises a `ValueError` if `use_abstract_mesh` is entered directly while an
  ambient `AbstractMesh` of a different size is already active. Entering an
  empty `AbstractMesh((), ())` first avoids that collision and cleanly restores
  the outer mesh on exit.

  Args:
    mesh: Target `AbstractMesh` to activate for the duration of the context.

  Yields:
    None.
  """
  prev = jax.sharding.get_abstract_mesh()
  if not prev.empty and not mesh.empty and prev.size != mesh.size:
    with (
        jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())),
        jax.sharding.use_abstract_mesh(mesh),
    ):
      yield
  else:
    with jax.sharding.use_abstract_mesh(mesh):
      yield


@contextlib.contextmanager
def cpu_interpret_mesh_context(cpu_mesh: jax.sharding.Mesh | None):
  """Binds `cpu_mesh` (or clears an ambient `AbstractMesh`) during CPU execution.

  Args:
    cpu_mesh: Replacement CPU mesh to bind, or `None` to clear an ambient mesh.

  Yields:
    None.
  """
  ctx_abs = jax.sharding.get_abstract_mesh()
  set_mesh_fn = getattr(jax, "set_mesh", None)
  if set_mesh_fn is not None and not ctx_abs.empty:
    # `jax.set_mesh` rejects meshes with manual axes; strip `axis_types` if any
    # axis is manual while keeping the concrete CPU device assignment.
    ctx_mesh = (
        jax.sharding.Mesh(cpu_mesh.devices, cpu_mesh.axis_names)
        if cpu_mesh is not None and getattr(cpu_mesh, "_any_axis_manual", False)
        else cpu_mesh
    )
    with set_mesh_fn(ctx_mesh):
      yield
  else:
    yield


def _unwrap_jax_jit(fn: Any) -> Any:
  """Unwraps `jax.jit` (`PjitFunction`) layers without stripping `shard_map`."""
  seen: set[int] = set()
  while hasattr(fn, "_fun") and hasattr(fn, "lower") and id(fn) not in seen:
    seen.add(id(fn))
    fn = getattr(fn, "_fun")
  return fn


def _matches_mesh_topology(mesh: MeshLike, target: MeshLike) -> bool:
  """Returns whether `mesh` and `target` have identical `(axis_name, size)` pairs."""
  return list(mesh.shape.items()) == list(target.shape.items())


def _is_hashable_partial(obj: Any) -> bool:
  """Returns whether `obj` is JAX's internal `HashablePartial` wrapper."""
  return (
      type(obj).__name__ == "HashablePartial"
      and hasattr(obj, "f")
      and hasattr(obj, "args")
      and hasattr(obj, "kwargs")
  )


def _get_ambient_jax_mesh() -> MeshLike | None:
  """Returns the currently active concrete or abstract JAX mesh, if any."""
  phys_mesh = jax._src.mesh.thread_resources.env.physical_mesh
  if not phys_mesh.empty:
    return phys_mesh
  get_mesh_fn = getattr(jax.sharding, "get_mesh", None)
  if get_mesh_fn is not None:
    try:
      pub_mesh = get_mesh_fn()
    except ValueError:
      pub_mesh = None
    if pub_mesh is not None and not pub_mesh.empty:
      return pub_mesh
  conc_fn = getattr(jax._src.mesh, "get_concrete_mesh", None)
  if conc_fn is not None:
    conc_mesh = conc_fn()
    if not conc_mesh.empty:
      return conc_mesh
  abs_mesh = jax.sharding.get_abstract_mesh()
  if not abs_mesh.empty:
    return abs_mesh
  return None


def find_function_mesh(fn: Any) -> MeshLike | None:
  """Finds a non-empty `Mesh` or `AbstractMesh` bound in `fn` or JAX context.

  Args:
    fn: Callable whose closures, defaults, partials, or ambient JAX context may
      bind a device mesh.

  Returns:
    The first non-empty `Mesh` or `AbstractMesh` discovered, or `None`.
  """
  visited: set[int] = set()

  def _search(obj: Any) -> MeshLike | None:
    if isinstance(obj, (jax.sharding.Mesh, jax.sharding.AbstractMesh)):
      return obj if obj.size > 0 else None
    if isinstance(obj, jax.sharding.NamedSharding):
      return obj.mesh if obj.mesh.size > 0 else None

    obj_id = id(obj)
    if obj_id in visited:
      return None
    visited.add(obj_id)

    # Check shardings attached to a `jax.jit` (`PjitFunction`) wrapper.
    if hasattr(obj, "_fun") and hasattr(obj, "lower"):
      jit_info = getattr(obj, "_jit_info", None)
      if jit_info is not None:
        for leaves_attr in ("in_shardings_leaves", "out_shardings_leaves"):
          found = _search(getattr(jit_info, leaves_attr, ()))
          if found is not None:
            return found
      return _search(getattr(obj, "_fun"))

    if _is_hashable_partial(obj):
      return (
          _search(getattr(obj, "f"))
          or _search(getattr(obj, "args"))
          or _search(getattr(obj, "kwargs"))
      )

    if isinstance(obj, (tuple, list, set, frozenset)):
      for item in obj:
        found = _search(item)
        if found is not None:
          return found
      return None

    if isinstance(obj, dict):
      for item in obj.values():
        found = _search(item)
        if found is not None:
          return found
      return None

    for child in _iter_callable_children(obj):
      found = _search(child)
      if found is not None:
        return found
    return None

  return _search(fn) or _get_ambient_jax_mesh()


def _replace_mesh_if_mock(
    val: Any,
    mesh_replacement: MeshLike,
    memo: dict[int, tuple[Any, bool]] | None = None,
    *,
    clone_closures: bool = False,
    keep_alive: list[Any] | None = None,
) -> tuple[Any, bool]:
  """Recursively replaces matching non-CPU meshes in values and closures.

  When verifying a multi-device kernel on CPU (or lowering an `AbstractMesh` op
  with `abstract_device=None` for TPU), any closed-over or partial-bound mesh
  matching `mesh_replacement`'s axis topology is swapped for `mesh_replacement`.

  Args:
    val: Object, pytree, or callable to rewrite.
    mesh_replacement: Target `Mesh` or `AbstractMesh` with matching axes/shape.
    memo: Optional `id(obj) -> (rewritten_obj, did_change)` memoization table.
    clone_closures: If `True`, allocates new `types.CellType` closure cells on
      functions so JAX/Pallas trace caches do not reuse traces from the old
      mesh.
    keep_alive: Optional list retaining temporary objects while `memo` is live
      so Python does not reuse their `id()` addresses during traversal.

  Returns:
    A `(rewritten_val, did_change)` pair.
  """
  if memo is None:
    memo = {}
  if keep_alive is None:
    keep_alive = []

  def _rewrite_seq(items: Sequence[Any]) -> tuple[list[Any], bool]:
    out: list[Any] = []
    changed = False
    for elem in items:
      new_elem, elem_changed = _rewrite(elem)
      out.append(new_elem)
      changed = changed or elem_changed
    return out, changed

  def _rewrite_dict(mapping: Mapping[Any, Any]) -> tuple[dict[Any, Any], bool]:
    out: dict[Any, Any] = {}
    changed = False
    for k, elem in mapping.items():
      new_elem, elem_changed = _rewrite(elem)
      out[k] = new_elem
      changed = changed or elem_changed
    return out, changed

  def _rewrite_pjit(pjit_fn: Any) -> tuple[Any, bool]:
    inner, changed = _rewrite(getattr(pjit_fn, "_fun"))
    jit_info = getattr(pjit_fn, "_jit_info", None)
    jit_kwargs: dict[str, Any] = {}
    if jit_info is not None:
      for attr in (
          "static_argnums",
          "static_argnames",
          "donate_argnums",
          "donate_argnames",
      ):
        attr_val = getattr(jit_info, attr, ())
        if attr_val:
          jit_kwargs[attr] = attr_val
      if getattr(jit_info, "keep_unused", False):
        jit_kwargs["keep_unused"] = True
      in_leaves = getattr(jit_info, "in_shardings_leaves", ())
      in_treedef = getattr(jit_info, "in_shardings_treedef", None)
      if (
          getattr(jit_info, "user_specified_in_shardings", False)
          and in_treedef is not None
      ):
        new_in_leaves, in_changed = _rewrite(in_leaves)
        changed = changed or in_changed
        jit_kwargs["in_shardings"] = in_treedef.unflatten(new_in_leaves)
      out_leaves = getattr(jit_info, "out_shardings_leaves", ())
      out_treedef = getattr(jit_info, "out_shardings_treedef", None)
      if out_treedef is not None and not (
          len(out_leaves) == 1
          and type(out_leaves[0]).__name__ == "UnspecifiedValue"
      ):
        new_out_leaves, out_changed = _rewrite(out_leaves)
        changed = changed or out_changed
        jit_kwargs["out_shardings"] = out_treedef.unflatten(new_out_leaves)
    return (jax.jit(inner, **jit_kwargs), True) if changed else (pjit_fn, False)

  def _rewrite_function(fn: types.FunctionType) -> tuple[Any, bool]:
    # Pre-seed `memo` before recursing into closure cells to break cycles in
    # recursive functions.
    fn_id = id(fn)
    memo[fn_id] = (fn, False)
    changed = False

    new_cells = None
    if fn.__closure__:
      cells_list = []
      for cell in fn.__closure__:
        try:
          contents = cell.cell_contents
        except ValueError:
          cells_list.append(cell)
          continue
        replaced, did_change = _rewrite(contents)
        if did_change or clone_closures:
          cells_list.append(types.CellType(replaced))
          changed = True
        else:
          cells_list.append(cell)
      new_cells = tuple(cells_list)

    new_defaults = fn.__defaults__
    if fn.__defaults__:
      defs_list, defs_changed = _rewrite_seq(fn.__defaults__)
      new_defaults = tuple(defs_list)
      changed = changed or defs_changed

    new_kwdefaults = fn.__kwdefaults__
    if fn.__kwdefaults__:
      new_kwdefaults, kwdefs_changed = _rewrite_dict(fn.__kwdefaults__)
      changed = changed or kwdefs_changed

    if not changed:
      return fn, False
    rebound = types.FunctionType(
        fn.__code__, fn.__globals__, fn.__name__, new_defaults, new_cells
    )
    rebound.__kwdefaults__ = new_kwdefaults
    rebound.__annotations__ = dict(fn.__annotations__)
    rebound.__qualname__ = fn.__qualname__
    return rebound, True

  def _rewrite(obj: Any) -> tuple[Any, bool]:
    # Base cases: Mesh, AbstractMesh, NamedSharding, and sharded ShapeDtypeStruct.
    if isinstance(obj, (jax.sharding.Mesh, jax.sharding.AbstractMesh)):
      if (
          obj is not mesh_replacement
          and is_mock_tpu_mesh(obj)
          and _matches_mesh_topology(obj, mesh_replacement)
      ):
        return mesh_replacement, True
      return obj, False

    if isinstance(obj, jax.sharding.NamedSharding):
      if (
          obj.mesh is not mesh_replacement
          and is_mock_tpu_mesh(obj.mesh)
          and _matches_mesh_topology(obj.mesh, mesh_replacement)
      ):
        return jax.sharding.NamedSharding(mesh_replacement, obj.spec), True
      return obj, False

    if isinstance(obj, jax.ShapeDtypeStruct) and isinstance(
        getattr(obj, "sharding", None), jax.sharding.NamedSharding
    ):
      replaced_sharding, changed = _rewrite(obj.sharding)
      if changed:
        return (
            jax.ShapeDtypeStruct(
                obj.shape, obj.dtype, sharding=replaced_sharding
            ),
            True,
        )
      return obj, False

    obj_id = id(obj)
    if obj_id in memo:
      return memo[obj_id]

    if hasattr(obj, "_fun") and hasattr(obj, "lower"):
      res = _rewrite_pjit(obj)
      memo[obj_id] = res
      return res

    if isinstance(obj, functools.partial):
      new_func, f_changed = _rewrite(obj.func)
      new_args, a_changed = _rewrite_seq(obj.args)
      new_kw, kw_changed = _rewrite_dict(obj.keywords or {})
      changed = f_changed or a_changed or kw_changed
      res = (
          (functools.partial(new_func, *new_args, **new_kw), True)
          if changed
          else (obj, False)
      )
      memo[obj_id] = res
      return res

    if _is_hashable_partial(obj):
      as_partial = functools.partial(
          getattr(obj, "f"),
          *getattr(obj, "args"),
          **(getattr(obj, "kwargs") or {}),
      )
      keep_alive.append(as_partial)
      rebound_p, changed = _rewrite(as_partial)
      res = (rebound_p, True) if changed else (obj, False)
      memo[obj_id] = res
      return res

    if isinstance(obj, types.FunctionType):
      res = _rewrite_function(obj)
      memo[obj_id] = res
      return res

    if type(obj) is tuple:  # pylint: disable=unidiomatic-typecheck
      items, changed = _rewrite_seq(obj)
      return (tuple(items), True) if changed else (obj, False)

    if type(obj) is list:  # pylint: disable=unidiomatic-typecheck
      items, changed = _rewrite_seq(obj)
      return (items, True) if changed else (obj, False)

    if type(obj) is dict:  # pylint: disable=unidiomatic-typecheck
      out_dict, changed = _rewrite_dict(obj)
      return (out_dict, True) if changed else (obj, False)

    return obj, False

  return _rewrite(val)


def replace_mock_meshes_in_pytree(
    tree: Any, mesh_replacement: MeshLike | None
) -> Any:
  """Replaces matching non-CPU meshes and shardings in `tree`."""
  if mesh_replacement is None:
    return tree
  return jax.tree.map(
      lambda leaf: _replace_mesh_if_mock(leaf, mesh_replacement)[0],
      tree,
  )


def rebind_function_meshes(
    fn: Callable[..., Any],
    mesh_replacement: MeshLike,
    *,
    clone_closures: bool = False,
) -> Callable[..., Any]:
  """Replaces closed-over, default, or partial-bound non-CPU meshes in `fn`.

  Args:
    fn: Target JAX callable (optionally wrapped in `jax.jit`).
    mesh_replacement: Replacement `Mesh` or `AbstractMesh` with matching axes.
    clone_closures: Whether to clone function closure cells to force fresh JAX
      and Pallas trace caches.

  Returns:
    A callable unwrapped past outer `jax.jit` with matching meshes rebound.
  """
  unwrapped = _unwrap_jax_jit(fn)
  rebound, _ = _replace_mesh_if_mock(
      unwrapped, mesh_replacement, clone_closures=clone_closures
  )
  return rebound


def unwrap_raw_jax_fn(
    op_or_fn: Any,
    *,
    prefer_cpu_mesh: bool = False,
    preserve_axis_types: bool = True,
) -> tuple[Callable[..., Any], MeshLike | None]:
  """Returns `(raw_jax_fn, active_mesh)` unwrapped past outer `jax.jit`.

  Args:
    op_or_fn: A `@jax_op`, `JaxCallable`, or raw JAX function.
    prefer_cpu_mesh: If `True` and a non-CPU mesh is bound to `op_or_fn`,
      replaces it with a matching virtual-CPU `jax.sharding.Mesh` (when enough
      virtual CPU devices exist) and rebinds closed-over mesh references in
      `raw_jax_fn`.
    preserve_axis_types: Whether to preserve `mesh.axis_types` on the created
      CPU mesh.

  Returns:
    A `(raw_jax_fn, mesh)` pair.
  """
  jax_callable = extract_jax_callable(op_or_fn)
  if jax_callable is not None:
    raw_fn = _unwrap_jax_jit(jax_callable.jit_fn)
    mesh: MeshLike | None = jax_callable.mesh
  else:
    raw_fn = _unwrap_jax_jit(op_or_fn)
    mesh = find_function_mesh(raw_fn) if prefer_cpu_mesh else None

  if prefer_cpu_mesh and mesh is not None and is_mock_tpu_mesh(mesh):
    cpu_mesh = make_cpu_mesh_for(mesh, preserve_axis_types=preserve_axis_types)
    if cpu_mesh is not None:
      raw_fn = rebind_function_meshes(raw_fn, cpu_mesh, clone_closures=True)
      return raw_fn, cpu_mesh

  return raw_fn, mesh


# ==============================================================================
# 4. `@jax_op` Contract & Host TPU Lowering Verification
# ==============================================================================


def _tensor_storage_ptr(tensor: torch.Tensor) -> int | None:
  """Returns the backing storage pointer for a non-empty non-meta tensor."""
  if tensor.device.type == "meta" or tensor.numel() == 0:
    return None
  base_ptr = tensor.untyped_storage().data_ptr()
  return base_ptr if base_ptr != 0 else None


def _eager_tensors_alias(a: torch.Tensor, b: torch.Tensor) -> bool:
  """Returns True if two eager tensors share the same backing storage buffer.

  TorchTPU donates the base XLA buffer when any view is donated, invalidating
  all views backed by the same `untyped_storage()`.

  Args:
    a: First eager tensor.
    b: Second eager tensor.

  Returns:
    `True` if `a` and `b` share a non-empty backing storage allocation.
  """
  if a is b:
    return True
  ptr_a = _tensor_storage_ptr(a)
  ptr_b = _tensor_storage_ptr(b)
  return ptr_a is not None and ptr_a == ptr_b


def _iter_spec_axis_names(spec: Any) -> list[str]:
  """Extracts all mesh axis names referenced inside a JAX `PartitionSpec`."""
  names: list[str] = []
  if spec is None:
    return names
  for entry in spec:
    if entry is None:
      continue
    if isinstance(entry, str):
      names.append(entry)
    elif isinstance(entry, (tuple, list)):
      names.extend(sub for sub in entry if isinstance(sub, str))
  return names


def _collect_jax_callable_tensor_args(
    jax_callable: pallas_lib.JaxCallable,
    norm_args: Sequence[Any],
    norm_kwargs: Mapping[str, Any],
) -> list[tuple[int, str, torch.Tensor]]:
  """Collects `(param_idx, label, tensor)` for all non-static tensor inputs."""
  collected: list[tuple[int, str, torch.Tensor]] = []
  # pylint: disable=protected-access
  for i, arg in enumerate(norm_args):
    if is_static_positional_arg(jax_callable, i):
      continue
    param = (
        jax_callable._orig_params[i]
        if i < len(jax_callable._orig_params)
        and jax_callable._orig_params[i].kind
        != inspect.Parameter.VAR_POSITIONAL
        else None
    )
    param_name = param.name if param is not None else f"arg_{i}"
    if isinstance(arg, torch.Tensor):
      collected.append((i, param_name, arg))
    else:
      for sub_idx, elem in enumerate(torch_pytree.tree_leaves(arg)):
        if isinstance(elem, torch.Tensor):
          collected.append((i, f"{param_name}[{sub_idx}]", elem))

  for k, arg in norm_kwargs.items():
    if _is_static_keyword_arg(jax_callable, k):
      continue
    param_idx = jax_callable._param_name_to_idx.get(k, -1)
    if isinstance(arg, torch.Tensor):
      collected.append((param_idx, k, arg))
    else:
      for sub_idx, elem in enumerate(torch_pytree.tree_leaves(arg)):
        if isinstance(elem, torch.Tensor):
          collected.append((param_idx, f"{k}[{sub_idx}]", elem))
  # pylint: enable=protected-access
  return collected


def _check_donor_aliasing_and_parameters(
    jax_callable: pallas_lib.JaxCallable,
    norm_args: Sequence[Any],
    norm_kwargs: Mapping[str, Any],
    op_name: str,
) -> list[findings.VerificationFinding]:
  """Checks eager storage aliasing on donated inputs and `nn.Parameter` donation."""
  detected: list[findings.VerificationFinding] = []
  donated_set = _get_donated_arg_indices(jax_callable, len(norm_args))
  if not donated_set:
    return detected
  invalidating_donated_set = _get_donated_arg_indices(
      jax_callable, len(norm_args), include_io_aliases=False
  )

  tensor_entries = _collect_jax_callable_tensor_args(
      jax_callable, norm_args, norm_kwargs
  )

  # 1. Check whether any donated tensor shares storage with another input.
  for idx_a, (p_a, name_a, ten_a) in enumerate(tensor_entries):
    for p_b, name_b, ten_b in tensor_entries[idx_a + 1 :]:
      if (p_a in donated_set or p_b in donated_set) and _eager_tensors_alias(
          ten_a, ten_b
      ):
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.ALIASED_DONOR_INPUTS,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"inputs {name_a!r} (arg {p_a}) and {name_b!r} (arg {p_b})"
                    " alias the same storage while at least one is donated via"
                    f" donate_argnums={tuple(sorted(donated_set))}"
                ),
            )
        )

  # 2. Check whether any invalidated donor tensor is an `nn.Parameter`.
  # Note: `input_output_aliases` copies the output back into the input tensor
  # rather than invalidating its PJRT buffer (`pallas.py:1376-1377`).
  for param_idx, name, tensor in tensor_entries:
    if param_idx in invalidating_donated_set and isinstance(
        tensor, torch.nn.Parameter
    ):
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.DONATED_PARAMETER,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"input {name!r} (arg {param_idx}) is an nn.Parameter but is"
                  " donated; the kernel may reuse its buffer and corrupt the"
                  " parameter, so pass a clone() or stop donating it"
              ),
          )
      )
  return detected


def _validate_partition_spec_entry(
    spec: Any,
    ndim: int,
    mesh: MeshLike,
    label: str,
    op_name: str,
    *,
    global_shape: Sequence[int] | None = None,
) -> list[findings.VerificationFinding]:
  """Checks rank, axis existence, uniqueness, and divisibility of one `PartitionSpec`."""
  detected: list[findings.VerificationFinding] = []
  if spec is None:
    return detected
  spec_tuple = tuple(spec)
  if len(spec_tuple) > ndim:
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_CONTRACTS,
            op_name=op_name,
            message=(
                f"{label} PartitionSpec {spec!r} has length {len(spec_tuple)},"
                f" exceeding tensor rank {ndim}"
            ),
        )
    )
    return detected

  axis_sizes = {ax: int(mesh.shape[ax]) for ax in mesh.axis_names}
  seen_axes: set[str] = set()
  unconstrained = getattr(jax.sharding.PartitionSpec, "UNCONSTRAINED", None)
  for dim_idx, entry in enumerate(spec_tuple):
    if entry is None or (unconstrained is not None and entry is unconstrained):
      continue
    axes = (
        (entry,)
        if isinstance(entry, str)
        else tuple(ax for ax in entry if isinstance(ax, str))
        if isinstance(entry, (tuple, list))
        else ()
    )
    shard_factor = 1
    for ax in axes:
      if ax not in axis_sizes:
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"{label} PartitionSpec {spec!r} references unknown mesh"
                    f" axis {ax!r}; available mesh axes are"
                    f" {tuple(mesh.axis_names)}"
                ),
            )
        )
      else:
        shard_factor *= axis_sizes[ax]
      if ax in seen_axes:
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"{label} PartitionSpec {spec!r} uses mesh axis {ax!r}"
                    " multiple times across tensor dimensions"
                ),
            )
        )
      seen_axes.add(ax)

    # Note: In TorchTPU's `@jax_op`, input tensors are per-device local shards
    # (multiplied by `shard_factor` to form the global JAX shape), whereas
    # output `aval.shape`s from JAX are global shapes that `torch_placeholder`
    # divides by `shard_factor` (`get_local_shape`). Thus `global_shape`
    # divisibility is checked on output `aval`s.
    if (
        global_shape is not None
        and dim_idx < len(global_shape)
        and shard_factor > 1
        and isinstance(global_shape[dim_idx], int)
        and global_shape[dim_idx] % shard_factor != 0
    ):
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"{label} global dimension {dim_idx} of size"
                  f" {global_shape[dim_idx]} is not divisible by mesh shard"
                  f" factor {shard_factor} for PartitionSpec {spec!r}"
              ),
          )
      )
  return detected


def _trailing_args_match_static_defaults(
    jax_callable: pallas_lib.JaxCallable,
    norm_args: Sequence[Any],
    start_idx: int,
) -> bool:
  """Returns whether trailing `norm_args[start_idx:]` match static defaults."""
  # pylint: disable=protected-access
  orig_params = jax_callable._orig_params
  # pylint: enable=protected-access
  if start_idx >= len(norm_args):
    return False
  for idx in range(start_idx, len(norm_args)):
    if not is_static_positional_arg(jax_callable, idx):
      return False
    if idx >= len(orig_params):
      return False
    default_val = orig_params[idx].default
    if default_val is inspect.Parameter.empty:
      return False
    try:
      matches = bool(norm_args[idx] == default_val)
    except Exception:  # pylint: disable=broad-exception-caught
      matches = norm_args[idx] is default_val
    if not matches:
      return False
  return True


def _check_mesh_and_input_specs(
    jax_callable: pallas_lib.JaxCallable,
    norm_args: Sequence[Any],
    op_name: str,
) -> tuple[bool, list[findings.VerificationFinding]]:
  """Validates `mesh` device IDs and `input_partition_specs` against `norm_args`."""
  detected: list[findings.VerificationFinding] = []
  mesh = jax_callable.mesh
  in_specs = jax_callable.input_partition_specs
  out_specs = jax_callable.output_partition_specs

  if mesh is None:
    if in_specs is not None or out_specs is not None:
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  "partition specs were provided without a `mesh` on the Pallas"
                  " op"
              ),
          )
      )
    return True, detected

  mesh_ok = True
  if not isinstance(mesh, jax.sharding.AbstractMesh):
    device_ids = [
        int(getattr(d, "id", idx)) for idx, d in enumerate(mesh.devices.flat)
    ]
    if len(set(device_ids)) != len(device_ids):
      mesh_ok = False
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"mesh contains duplicate device IDs {device_ids}; every"
                  " mesh entry must map to a distinct physical device"
              ),
          )
      )

  if in_specs is None:
    return mesh_ok, detected

  # Catch passing a single `PartitionSpec` instead of a sequence of specs.
  if isinstance(in_specs, jax.sharding.PartitionSpec):
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_CONTRACTS,
            op_name=op_name,
            message=(
                "input_partition_specs must be a sequence of PartitionSpecs"
                f" (one per positional argument), got a bare {in_specs!r}"
            ),
        )
    )
    return False, detected

  num_non_static = sum(
      1
      for i in range(len(norm_args))
      if not is_static_positional_arg(jax_callable, i)
  )
  if len(in_specs) != len(norm_args) and not (
      len(in_specs) < len(norm_args)
      and _trailing_args_match_static_defaults(
          jax_callable, norm_args, len(in_specs)
      )
  ):
    # `pallas.jax_placeholders` zips `norm_args` with `input_partition_specs`
    # without `strict=True`, so omitting `None`/`PartitionSpec()` entries for
    # non-defaulted static positional arguments silently drops trailing args!
    if len(in_specs) == num_non_static and num_non_static != len(norm_args):
      detail = (
          f"input_partition_specs has length {len(in_specs)}, expected"
          f" {len(norm_args)} (one entry per positional argument, using"
          " None or PartitionSpec() for static non-tensor arguments;"
          " omitting entries truncates positional arguments)"
      )
    else:
      detail = (
          f"input_partition_specs has length {len(in_specs)}, expected"
          f" {len(norm_args)} positional argument(s)"
      )
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_CONTRACTS,
            op_name=op_name,
            message=detail,
        )
    )
    return False, detected

  for idx, (arg_val, spec) in enumerate(zip(norm_args, in_specs)):
    if is_static_positional_arg(jax_callable, idx):
      sharded_axes = _iter_spec_axis_names(spec)
      if sharded_axes:
        mesh_ok = False
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"input[{idx}] PartitionSpec {spec!r} shards mesh axes"
                    f" {tuple(sharded_axes)} on a static non-tensor argument"
                    " (use None or PartitionSpec())"
                ),
            )
        )
    elif isinstance(arg_val, torch.Tensor):
      spec_errs = _validate_partition_spec_entry(
          spec, arg_val.ndim, mesh, f"input[{idx}]", op_name
      )
      if spec_errs:
        mesh_ok = False
        detected.extend(spec_errs)
    elif isinstance(arg_val, (list, tuple)):
      for sub_idx, elem in enumerate(arg_val):
        if isinstance(elem, torch.Tensor):
          spec_errs = _validate_partition_spec_entry(
              spec, elem.ndim, mesh, f"input[{idx}][{sub_idx}]", op_name
          )
          if spec_errs:
            mesh_ok = False
            detected.extend(spec_errs)

  return mesh_ok, detected


def _infer_jax_callable_output_struct(
    jax_callable: pallas_lib.JaxCallable,
    norm_args: Sequence[Any],
    norm_kwargs: Mapping[str, Any],
    tpu_version: findings.TpuVersion,
) -> Any:
  """Infers abstract output shapes/dtypes of `jax_callable` via `jax.eval_shape`."""
  raw_fn, eval_mesh = unwrap_raw_jax_fn(jax_callable)
  # pylint: disable=protected-access
  jax_args, jax_kwargs = jax_callable._get_jax_placeholders(
      norm_args, dict(norm_kwargs)
  )
  # pylint: enable=protected-access

  # Partition `(jax_args, jax_kwargs)` into dynamic `ShapeDtypeStruct` leaves
  # (passed through `jax.eval_shape`) and static Python values (closed over),
  # so arbitrary nested tensor sequences and non-hashable static args both work.
  leaves, treedef = jax.tree.flatten((jax_args, jax_kwargs))
  dyn_structs = [
      leaf for leaf in leaves if isinstance(leaf, jax.ShapeDtypeStruct)
  ]

  def _shape_fn(*traced_leaves: Any) -> Any:
    it = iter(traced_leaves)
    reconstructed = [
        next(it) if isinstance(leaf, jax.ShapeDtypeStruct) else leaf
        for leaf in leaves
    ]
    call_args, call_kwargs = treedef.unflatten(reconstructed)
    return raw_fn(*call_args, **call_kwargs)

  try:
    return jax.eval_shape(_shape_fn, *dyn_structs)
  except ValueError as exc:
    if not _is_missing_tpu_device_kind_error(exc) or eval_mesh is not None:
      raise
    with override_abstract_mesh(abstract_tpu_mesh(tpu_version)):
      return jax.eval_shape(_shape_fn, *dyn_structs)


def verify_op_contracts(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    tpu_version: findings.TpuVersion | None = None,
) -> findings.VerificationReport:
  """Checks static and eager contracts on a TorchTPU Pallas custom op."""
  resolved_tpu_version = infer_tpu_version(op_or_fn, tpu_version)
  kwargs = dict(example_kwargs or {})
  op_name = get_op_display_name(op_or_fn)

  jax_callable = extract_jax_callable(op_or_fn)
  if jax_callable is None:
    return findings.VerificationReport()

  # pylint: disable=protected-access
  norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
      example_args, kwargs
  )
  # pylint: enable=protected-access

  # 1. Check eager storage aliasing on donated inputs and `nn.Parameter` donation.
  detected = _check_donor_aliasing_and_parameters(
      jax_callable, norm_args, norm_kwargs, op_name
  )

  # 2. Check mesh device uniqueness & input PartitionSpec contracts.
  mesh_ok, mesh_findings = _check_mesh_and_input_specs(
      jax_callable, norm_args, op_name
  )
  detected.extend(mesh_findings)
  if not mesh_ok:
    return findings.report_of(_CONTRACTS, detected)

  # 3. Check zero-output silent skip, output PartitionSpecs, and register_fake.
  mesh = jax_callable.mesh
  out_specs = jax_callable.output_partition_specs
  try:
    out_struct = _infer_jax_callable_output_struct(
        jax_callable, norm_args, norm_kwargs, resolved_tpu_version
    )
    out_avals = [
        leaf
        for leaf in jax.tree.leaves(out_struct)
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
    ]
    out_specs_ok = True
    if not out_avals:
      out_specs_ok = False
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.ZERO_OUTPUT_SILENT_SKIP,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  "jax_op returns zero tensor outputs; PyCallCustomKernel in"
                  " TorchTPU silently skips execution when output_shapes is"
                  " empty, dropping any side effects or barriers"
              ),
          )
      )
    elif mesh is not None and out_specs is not None:
      # pylint: disable=protected-access
      norm_out_specs = pallas_lib._normalize_output_partition_specs(
          out_specs, len(out_avals)
      )
      # pylint: enable=protected-access
      if norm_out_specs is not None and len(norm_out_specs) != len(out_avals):
        out_specs_ok = False
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"output_partition_specs has length {len(norm_out_specs)},"
                    f" expected {len(out_avals)}"
                ),
            )
        )
      elif norm_out_specs is not None:
        for out_idx, (aval, spec) in enumerate(zip(out_avals, norm_out_specs)):
          spec_errs = _validate_partition_spec_entry(
              spec,
              len(aval.shape),
              mesh,
              f"output[{out_idx}]",
              op_name,
              global_shape=aval.shape,
          )
          if spec_errs:
            out_specs_ok = False
            detected.extend(spec_errs)

    if out_specs_ok:
      detected.extend(
          _verify_fake_kernel_outputs(
              op_or_fn,
              jax_callable,
              example_args,
              kwargs,
              out_struct,
              op_name,
              tpu_version=resolved_tpu_version,
          )
      )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    reported_axis = False
    if mesh is not None and out_specs is not None:
      # pylint: disable=protected-access
      fallback_out_specs = (
          pallas_lib._normalize_output_partition_specs(out_specs, 1) or ()
      )
      # pylint: enable=protected-access
      for out_idx, spec in enumerate(fallback_out_specs):
        for ax in _iter_spec_axis_names(spec):
          if ax not in mesh.axis_names:
            reported_axis = True
            detected.append(
                findings.VerificationFinding(
                    category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                    severity=findings.Severity.ERROR,
                    phase=_CONTRACTS,
                    op_name=op_name,
                    message=(
                        f"output[{out_idx}] PartitionSpec {spec!r} references"
                        f" unknown mesh axis {ax!r} ({exc})"
                    ),
                )
            )
    if not reported_axis:
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
              severity=findings.Severity.INFO,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  "skipped the output checks because the output shapes could"
                  f" not be inferred ({type(exc).__name__}:"
                  f" {_truncate(str(exc), 300)})"
              ),
          )
      )

  return findings.report_of(_CONTRACTS, detected)


def _is_default_jax_op_fake(
    op_def: Any, jax_callable: pallas_lib.JaxCallable
) -> bool:
  """Returns whether `op_def` uses `jax_op`'s unmodified built-in `fake_fn`."""
  abstract_fn = getattr(op_def, "_abstract_fn", None)
  if not isinstance(abstract_fn, types.FunctionType):
    return False
  if (
      abstract_fn.__module__ != pallas_lib.__name__
      or abstract_fn.__name__ != "fake_fn"
  ):
    return False
  nonlocals_map = inspect.getclosurevars(abstract_fn).nonlocals
  return (
      nonlocals_map.get("wrapped_fn") is jax_callable
      and nonlocals_map.get("mesh") is jax_callable.mesh
      and nonlocals_map.get("output_partition_specs")
      == jax_callable.output_partition_specs
  )


def _verify_fake_kernel_outputs(
    op_or_fn: Any,
    jax_callable: pallas_lib.JaxCallable,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any],
    out_struct: Any,
    op_name: str,
    *,
    tpu_version: findings.TpuVersion = DEFAULT_TPU_VERSION,
) -> list[findings.VerificationFinding]:
  """Compares custom `register_fake` outputs with `JaxCallable` runtime outputs."""
  op_def = resolve_custom_op_def(op_or_fn)
  if op_def is None or _is_default_jax_op_fake(op_def, jax_callable):
    return []
  if getattr(op_def, "_abstract_fn", None) is None:
    return [
        findings.VerificationFinding(
            category=findings.FindingCategory.FAKE_KERNEL_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_CONTRACTS,
            op_name=op_name,
            message=(
                "custom op has no register_fake implementation, so"
                " torch.compile and FX tracing cannot infer output metadata"
            ),
        )
    ]

  mesh = jax_callable.mesh
  out_specs = jax_callable.output_partition_specs
  out_leaves, out_tree = jax.tree.flatten(out_struct)
  # pylint: disable=protected-access
  specs = pallas_lib._normalize_output_partition_specs(
      out_specs, len(out_leaves)
  ) or [None] * len(out_leaves)
  # pylint: enable=protected-access

  # `jax.export` produces plain `ShapedArray`s without sharding metadata on
  # `lowered.out_avals`, so `JaxCallable.__call__` sizes runtime output buffers
  # solely from `mesh` and `output_partition_specs`.
  runtime_leaves = [
      pallas_lib.torch_placeholder(
          jax.core.ShapedArray(leaf.shape, leaf.dtype)
          if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
          else leaf,
          mesh=mesh,
          partition_spec=spec,
          device="meta",
      )
      for leaf, spec in zip(out_leaves, specs)
  ]

  if mesh is None:
    target: Any = override_abstract_mesh(abstract_tpu_mesh(tpu_version))
  elif (
      isinstance(mesh, jax.sharding.AbstractMesh)
      and mesh.abstract_device is None
  ):
    target = override_abstract_mesh(
        mesh.update(abstract_device=abstract_tpu_device(tpu_version))
    )
  else:
    target = contextlib.nullcontext()

  try:
    with fake_tensor.FakeTensorMode(allow_non_fake_inputs=True), target:

      def _to_fake(val: Any) -> Any:
        if not isinstance(val, torch.Tensor):
          return val
        dev = "cpu" if val.device.type == "meta" else val.device
        return torch.empty_strided(
            tuple(int(d) for d in val.shape),
            tuple(int(s) for s in val.stride()),
            dtype=val.dtype,
            device=dev,
        )

      fake_args = jax.tree.map(_to_fake, tuple(example_args))
      fake_kwargs = jax.tree.map(_to_fake, dict(example_kwargs))
      fake_out = op_def(*fake_args, **fake_kwargs)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    msg = _truncate(str(exc), 300)
    if _is_missing_tpu_device_kind_error(exc):
      return [
          findings.VerificationFinding(
              category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
              severity=findings.Severity.INFO,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  "skipped register_fake output check because the TPU"
                  " generation is unknown"
              ),
          )
      ]
    return [
        findings.VerificationFinding(
            category=findings.FindingCategory.FAKE_KERNEL_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_CONTRACTS,
            op_name=op_name,
            message=f"register_fake raised {type(exc).__name__}: {msg}",
        )
    ]

  fake_leaves, fake_tree = jax.tree.flatten(fake_out)
  if fake_tree != out_tree:
    return [
        findings.VerificationFinding(
            category=findings.FindingCategory.FAKE_KERNEL_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_CONTRACTS,
            op_name=op_name,
            message=(
                f"register_fake returned pytree {fake_tree}"
                f" ({len(fake_leaves)} leaf/leaves), whereas JaxCallable"
                f" allocates {out_tree} ({len(runtime_leaves)} leaf/leaves)"
                " at runtime"
            ),
        )
    ]

  detected: list[findings.VerificationFinding] = []
  for out_idx, (fake_leaf, rt_leaf) in enumerate(
      zip(fake_leaves, runtime_leaves)
  ):
    label = f"output[{out_idx}]"
    if isinstance(fake_leaf, torch.Tensor) != isinstance(rt_leaf, torch.Tensor):
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.FAKE_KERNEL_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"{label} register_fake returned {type(fake_leaf).__name__},"
                  " whereas JaxCallable allocates"
                  f" {type(rt_leaf).__name__} at runtime"
              ),
          )
      )
      continue
    if not isinstance(fake_leaf, torch.Tensor) or not isinstance(
        rt_leaf, torch.Tensor
    ):
      continue
    fake_shape = tuple(int(d) for d in fake_leaf.shape)
    rt_shape = tuple(int(d) for d in rt_leaf.shape)
    if fake_shape != rt_shape:
      if mesh is not None and out_specs is None:
        hint = (
            " (JaxCallable has mesh set with output_partition_specs=None, so"
            " runtime outputs are allocated at global shape)"
        )
      elif mesh is not None:
        hint = (
            f" (check output_partition_specs={out_specs!r} against"
            " register_fake)"
        )
      else:
        hint = ""
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.FAKE_KERNEL_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"{label} register_fake shape {fake_shape} does not match"
                  f" JaxCallable runtime output shape {rt_shape}{hint}"
              ),
          )
      )
    if fake_leaf.dtype != rt_leaf.dtype:
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.FAKE_KERNEL_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"{label} register_fake dtype {fake_leaf.dtype} does not"
                  f" match JaxCallable runtime output dtype {rt_leaf.dtype}"
              ),
          )
      )
  return detected


def _prepare_tpu_export(
    jax_callable: pallas_lib.JaxCallable,
    jax_args: list[Any],
    jax_kwargs: dict[str, Any],
    tpu_version: findings.TpuVersion,
) -> tuple[Any, list[Any], dict[str, Any], Any]:
  """Prepares `(export_fn, jax_args, jax_kwargs, mesh_context)` for TPU lowering."""
  export_fn = jax_callable.exported
  mesh = jax_callable.mesh

  # Case 1: Single-device op (`mesh is None`) — bind a 1-chip AbstractMesh
  # targeting `tpu_version` so `jax.export` knows the TPU generation.
  if mesh is None:
    return (
        export_fn,
        jax_args,
        jax_kwargs,
        override_abstract_mesh(abstract_tpu_mesh(tpu_version)),
    )

  # Case 2: Multi-device op with an `AbstractMesh` whose `abstract_device` was
  # left `None` — attach `abstract_tpu_device(tpu_version)` and rebind any
  # closed-over references to the mesh before exporting.
  if (
      isinstance(mesh, jax.sharding.AbstractMesh)
      and mesh.abstract_device is None
  ):
    tpu_mesh = mesh.update(abstract_device=abstract_tpu_device(tpu_version))
    jax_args = replace_mock_meshes_in_pytree(jax_args, tpu_mesh)
    jax_kwargs = replace_mock_meshes_in_pytree(jax_kwargs, tpu_mesh)
    rebound_raw = rebind_function_meshes(
        _unwrap_jax_jit(jax_callable.jit_fn), tpu_mesh, clone_closures=True
    )
    jit_kwargs: dict[str, Any] = {
        "static_argnums": jax_callable.static_argnums,
        "donate_argnums": jax_callable.donate_argnums,
        "keep_unused": True,
    }
    if jax_callable.static_argnames:
      jit_kwargs["static_argnames"] = jax_callable.static_argnames
    with jax._src.config.export_ignore_forward_compatibility(True):
      export_fn = jax.export.export(
          jax.jit(rebound_raw, **jit_kwargs),
          platforms=["tpu"],
          disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
      )
    ctx = (
        override_abstract_mesh(tpu_mesh)
        if not jax.sharding.get_abstract_mesh().empty
        else contextlib.nullcontext()
    )
    return export_fn, jax_args, jax_kwargs, ctx

  # Case 3: Concrete or mock `Mesh` (or `AbstractMesh` with `abstract_device`
  # already set) — the mesh's devices already specify the target TPU generation.
  return export_fn, jax_args, jax_kwargs, contextlib.nullcontext()


def verify_tpu_lowering(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    tpu_version: findings.TpuVersion | None = None,
) -> findings.VerificationReport:
  """Lowers a TorchTPU Pallas op for TPU on the host, without a TPU.

  This is the lowering TorchTPU performs the first time the op runs, so a
  failure here means the op fails on TPU too, for example because of
  unaligned block shapes or vector loads and stores on HBM refs.

  Args:
    op_or_fn: A `@jax_op` or `JaxCallable`.
    example_args: Example positional arguments.
    example_kwargs: Example keyword arguments.
    tpu_version: Target `TpuVersion` to lower for, or `None` to infer from the
      kernel. Ops with a mesh are lowered for the device kind of their mesh
      devices instead.

  Returns:
    A report with a `LOWERING_FAILURE` error if lowering fails.
  """
  resolved_tpu_version = infer_tpu_version(op_or_fn, tpu_version)
  jax_callable = extract_jax_callable(op_or_fn)
  if jax_callable is None:
    return findings.VerificationReport()
  op_name = get_op_display_name(op_or_fn)

  # pylint: disable=protected-access
  norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
      example_args, dict(example_kwargs or {})
  )
  jax_args, jax_kwargs = jax_callable._get_jax_placeholders(
      norm_args, norm_kwargs
  )
  # pylint: enable=protected-access
  export_fn, jax_args, jax_kwargs, target = _prepare_tpu_export(
      jax_callable, jax_args, jax_kwargs, resolved_tpu_version
  )

  try:
    # Matches the forward-compatibility mode TorchTPU lowers with.
    with target, jax._src.config.export_ignore_forward_compatibility(True):
      export_fn(*jax_args, **jax_kwargs)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    message = _truncate(f"{exc}", _MAX_LOWERING_MESSAGE_CHARS)
    if _is_missing_tpu_device_kind_error(exc):
      finding = findings.VerificationFinding(
          category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
          severity=findings.Severity.INFO,
          phase=_LOWERING,
          op_name=op_name,
          message=(
              "skipped TPU lowering because the TPU generation is unknown;"
              " build the mesh with torch.tpu.pallas.get_pallas_mesh or with"
              " devices whose device_kind names a TPU generation"
          ),
      )
      return findings.report_of(_LOWERING, [finding])
    target_desc = (
        resolved_tpu_version.value if jax_callable.mesh is None else "its mesh"
    )
    finding = findings.VerificationFinding(
        category=findings.FindingCategory.LOWERING_FAILURE,
        severity=findings.Severity.ERROR,
        phase=_LOWERING,
        op_name=op_name,
        message=(
            f"the kernel fails Pallas TPU lowering for {target_desc}, so it"
            f" cannot run on TPU ({type(exc).__name__}: {message})"
        ),
    )
    return findings.report_of(_LOWERING, [finding])
  return findings.report_of(_LOWERING)


# ==============================================================================
# 5. FX Graph Linear Buffer Donation & Aliasing Verification
# ==============================================================================

_GETATTR_VIEW_NAMES = frozenset({"T", "mT", "H", "real", "imag", "data"})

_VIEW_FALLBACK_NAMES = frozenset({
    "alias",
    "as_strided",
    "chunk",
    "detach",
    "diagonal",
    "expand",
    "expand_as",
    "flatten",
    "movedim",
    "narrow",
    "permute",
    "reshape",
    "select",
    "slice",
    "split",
    "split_with_sizes",
    "squeeze",
    "swapaxes",
    "swapdims",
    "t",
    "transpose",
    "unbind",
    "unflatten",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze",
    "view",
    "view_as",
    "_unsafe_view",
})

_METADATA_ONLY_FX_NAMES = frozenset({
    "dim",
    "empty_like",
    "full_like",
    "is_contiguous",
    "new_empty",
    "new_full",
    "new_ones",
    "new_zeros",
    "numel",
    "ones_like",
    "size",
    "storage_offset",
    "stride",
    "sym_numel",
    "sym_size",
    "sym_stride",
    "zeros_like",
})


def resolve_fx_attr(gm: fx.GraphModule, target_path: str) -> Any:
  """Resolves a dotted `get_attr` path on `gm`, returning `None` if absent."""
  curr: Any = gm
  for part in target_path.split("."):
    if not hasattr(curr, part):
      return None
    curr = getattr(curr, part)
  return curr


def _is_metadata_only_fx_node(node: fx.Node) -> bool:
  """Returns True if `node` only inspects tensor shape/dtype/device metadata."""
  if node.op not in ("call_function", "call_method"):
    return False
  target_name = getattr(node.target, "__name__", str(node.target)).split(".")[0]
  return target_name in _METADATA_ONLY_FX_NAMES


def _is_set_inplace_fx_node(node: fx.Node) -> bool:
  """Returns True if `node` is an in-place `Tensor.set_` storage rebind."""
  if not node.args or not isinstance(node.args[0], fx.Node):
    return False
  if node.op == "call_method":
    return str(node.target) == "set_"
  if node.op == "call_function":
    target_name = getattr(node.target, "__name__", str(node.target)).split(".")[
        0
    ]
    return target_name == "set_"
  return False


def iter_fx_nodes(value: Any) -> list[fx.Node]:
  """Collects all `fx.Node` leaves inside nested tuples, lists, dicts, or slices."""
  found: list[fx.Node] = []
  for leaf in torch_pytree.tree_leaves(value):
    if isinstance(leaf, fx.Node):
      found.append(leaf)
    elif isinstance(leaf, slice):
      for bound in (leaf.start, leaf.stop, leaf.step):
        if isinstance(bound, fx.Node):
          found.append(bound)
  return found


def _fx_op_per_return_aliases(
    target: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
) -> dict[int, list[fx.Node]] | None:
  """Maps tuple output indices to aliased input `fx.Node`s."""
  kwargs = kwargs or {}
  out_kw = kwargs.get("out")
  if isinstance(out_kw, (tuple, list)) and len(out_kw) > 1:
    per_ret = {idx: iter_fx_nodes(elem) for idx, elem in enumerate(out_kw)}
    if any(per_ret.values()):
      return per_ret

  schema = getattr(target, "_schema", None)
  if schema is None or len(schema.returns) <= 1:
    return None

  arg_by_alias_set: dict[frozenset[str], list[fx.Node]] = {}
  for idx, schema_arg in enumerate(schema.arguments):
    alias_info = getattr(schema_arg, "alias_info", None)
    before_set = frozenset(getattr(alias_info, "before_set", ()) or ())
    if not before_set or "*" in before_set:
      continue
    if idx < len(args):
      val = args[idx]
    elif schema_arg.name in kwargs:
      val = kwargs[schema_arg.name]
    else:
      continue
    nodes = iter_fx_nodes(val)
    if nodes:
      arg_by_alias_set.setdefault(before_set, []).extend(nodes)

  if not arg_by_alias_set:
    return None

  per_ret_schema: dict[int, list[fx.Node]] = {}
  for ret_idx, ret in enumerate(schema.returns):
    ret_alias = getattr(ret, "alias_info", None)
    ret_sets = frozenset(getattr(ret_alias, "before_set", ()) or ())
    if not ret_sets:
      continue
    matched: list[fx.Node] = []
    for aset, nodes in arg_by_alias_set.items():
      if aset & ret_sets:
        matched.extend(nodes)
    if matched:
      per_ret_schema[ret_idx] = matched
  return per_ret_schema or None


def fx_op_aliases_inputs(
    target: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
) -> list[fx.Node]:
  """Returns input `fx.Node`s whose storage may be viewed or mutated by `target`."""
  kwargs = kwargs or {}
  if target is getattr and len(args) >= 2:
    first, attr = args[0], args[1]
    if isinstance(first, fx.Node) and attr in _GETATTR_VIEW_NAMES:
      return [first]

  aliased_nodes: list[fx.Node] = []
  out_kw = kwargs.get("out")
  if out_kw is not None:
    aliased_nodes.extend(iter_fx_nodes(out_kw))

  schema = getattr(target, "_schema", None)
  if schema is not None:
    return_has_alias = any(
        getattr(ret, "alias_info", None) is not None for ret in schema.returns
    )
    if return_has_alias:
      for idx, schema_arg in enumerate(schema.arguments):
        if getattr(schema_arg, "alias_info", None) is None:
          continue
        if idx < len(args):
          val = args[idx]
        elif schema_arg.name in kwargs:
          val = kwargs[schema_arg.name]
        else:
          continue
        aliased_nodes.extend(iter_fx_nodes(val))
      if aliased_nodes:
        return aliased_nodes

  target_name = getattr(target, "__name__", str(target)).split(".")[0]
  if target_name in _VIEW_FALLBACK_NAMES or (
      target_name.endswith("_") and not target_name.startswith("_")
  ):
    first = args[0] if args else kwargs.get("input", kwargs.get("self"))
    if isinstance(first, fx.Node):
      aliased_nodes.append(first)
  return aliased_nodes


def _extract_root_nn_module(target: Any) -> torch.nn.Module | None:
  """Returns the backing `nn.Module` if `target` is a module or bound method."""
  if isinstance(target, torch.nn.Module):
    return target
  bound_self = getattr(target, "__self__", None)
  if isinstance(bound_self, torch.nn.Module):
    return bound_self
  return None


@contextlib.contextmanager
def _preserve_module_state(target: Any) -> Iterator[None]:
  """Snapshots and restores `nn.Module` attributes, buffers, and tensors."""
  root_mod = _extract_root_nn_module(target)
  if root_mod is None:
    yield
    return

  snapshots: list[
      tuple[
          torch.nn.Module,
          dict[str, Any],
          dict[str, torch.nn.Parameter | None],
          dict[str, torch.Tensor | None],
      ]
  ] = []
  storage_backups: list[
      tuple[torch.Tensor, Any, int, tuple[int, ...], tuple[int, ...]]
  ] = []
  data_backups: list[tuple[torch.Tensor, torch.Tensor]] = []
  seen_tensors: set[int] = set()

  for mod in root_mod.modules():
    # pylint: disable=protected-access
    params_copy = dict(mod._parameters)
    buffers_copy = dict(mod._buffers)
    # pylint: enable=protected-access
    dict_copy = dict(mod.__dict__)
    snapshots.append((mod, dict_copy, params_copy, buffers_copy))
    for t in (
        *params_copy.values(),
        *buffers_copy.values(),
        *dict_copy.values(),
    ):
      if (
          isinstance(t, torch.Tensor)
          and not isinstance(t, fake_tensor.FakeTensor)
          and t.device.type == "cpu"
          and id(t) not in seen_tensors
      ):
        seen_tensors.add(id(t))
        storage_backups.append((
            t,
            t.untyped_storage(),
            int(t.storage_offset()),
            tuple(t.size()),
            tuple(t.stride()),
        ))
        if t.numel() <= 65536:
          data_backups.append((t, t.detach().clone()))

  try:
    yield
  finally:
    for mod, dict_copy, params_copy, buffers_copy in snapshots:
      mod.__dict__.clear()
      mod.__dict__.update(dict_copy)
      # pylint: disable=protected-access
      mod._parameters.clear()
      mod._parameters.update(params_copy)
      mod._buffers.clear()
      mod._buffers.update(buffers_copy)
      # pylint: enable=protected-access
    with torch.no_grad():
      for t, orig_storage, orig_off, orig_size, orig_stride in storage_backups:
        if (
            t.untyped_storage().data_ptr() != orig_storage.data_ptr()
            or int(t.storage_offset()) != orig_off
            or tuple(t.size()) != orig_size
            or tuple(t.stride()) != orig_stride
        ):
          t.set_(orig_storage, orig_off, orig_size, orig_stride)
      for t, backup in data_backups:
        if t.shape == backup.shape and t.dtype == backup.dtype:
          t.copy_(backup)


def _make_thread_local_pallas_call(orig_pallas_call: Any) -> Any:
  """Returns a `pl.pallas_call` wrapper stubbing shapes on active threads."""

  def _thread_local_pallas_call(*p_args: Any, **p_kwargs: Any) -> Any:
    if int(getattr(_THREAD_PATCH_STATE, "stub_pallas_call_depth", 0)) <= 0:
      return orig_pallas_call(*p_args, **p_kwargs)
    out_shape = p_args[1] if len(p_args) > 1 else p_kwargs.get("out_shape")

    def _stub(*call_args: Any, **call_kwargs: Any) -> Any:
      del call_args, call_kwargs
      return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), out_shape)

    return _stub

  return _thread_local_pallas_call


@contextlib.contextmanager
def _stub_pallas_call_out_shapes() -> Iterator[None]:
  """Temporarily stubs `pl.pallas_call` on the current thread only."""
  global _PALLAS_CALL_STUB_REFCOUNT, _ORIG_PALLAS_CALL
  depth = int(getattr(_THREAD_PATCH_STATE, "stub_pallas_call_depth", 0))
  inc_refcount = False
  try:
    _THREAD_PATCH_STATE.stub_pallas_call_depth = depth + 1
    with _PATCH_LOCK:
      if _PALLAS_CALL_STUB_REFCOUNT == 0:
        orig_pallas_call = pl.pallas_call
        _ORIG_PALLAS_CALL = orig_pallas_call
        pl.pallas_call = _make_thread_local_pallas_call(orig_pallas_call)
      _PALLAS_CALL_STUB_REFCOUNT += 1
      inc_refcount = True
    yield
  finally:
    _THREAD_PATCH_STATE.stub_pallas_call_depth = depth
    if inc_refcount:
      with _PATCH_LOCK:
        _PALLAS_CALL_STUB_REFCOUNT -= 1
        if _PALLAS_CALL_STUB_REFCOUNT == 0 and _ORIG_PALLAS_CALL is not None:
          pl.pallas_call = _ORIG_PALLAS_CALL
          _ORIG_PALLAS_CALL = None


def _make_eval_shape_fallback_exported(
    jc: pallas_lib.JaxCallable, orig_exported: Any
) -> Any:
  """Returns a thread-gated `jc.exported` wrapper using `eval_shape`."""

  def _wrapped_exported(*j_args: Any, **j_kwargs: Any) -> Any:
    if int(getattr(_THREAD_PATCH_STATE, "eval_shape_fallback_depth", 0)) <= 0:
      return orig_exported(*j_args, **j_kwargs)
    try:
      return orig_exported(*j_args, **j_kwargs)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if (
          isinstance(exc, ValueError)
          and _is_missing_tpu_device_kind_error(exc)
          and jax.sharding.get_abstract_mesh().empty
      ):
        raise
      try:
        out_struct = jax.eval_shape(jc.jit_fn, *j_args, **j_kwargs)
      except Exception:  # pylint: disable=broad-exception-caught
        raw_fn, _ = unwrap_raw_jax_fn(jc)
        with _stub_pallas_call_out_shapes():
          out_struct = jax.eval_shape(raw_fn, *j_args, **j_kwargs)
      out_leaves, out_tree = jax.tree.flatten(out_struct)
      out_avals = [
          jax.core.ShapedArray(leaf.shape, leaf.dtype)
          if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
          else leaf
          for leaf in out_leaves
      ]
      return types.SimpleNamespace(out_avals=out_avals, out_tree=out_tree)

  return _wrapped_exported


@contextlib.contextmanager
def _eval_shape_fallback_for_jax_callables(roots: Any) -> Iterator[None]:
  """Infers `jax_op` output shapes on this thread when TPU lowering fails.

  The `JaxCallable.exported` wrapper is gated by `threading.local()` so
  concurrent calls on other threads invoke the original `exported` method
  unchanged.

  Args:
    roots: Root callable, args, and kwargs scanned for `JaxCallable` instances.

  Yields:
    None.
  """
  discovered: list[pallas_lib.JaxCallable] = []
  seen_jc_ids: set[int] = set()
  for obj in _collect_callable_objects(roots):
    if isinstance(obj, pallas_lib.JaxCallable) and id(obj) not in seen_jc_ids:
      seen_jc_ids.add(id(obj))
      discovered.append(obj)

  depth = int(getattr(_THREAD_PATCH_STATE, "eval_shape_fallback_depth", 0))
  installed: list[pallas_lib.JaxCallable] = []
  try:
    _THREAD_PATCH_STATE.eval_shape_fallback_depth = depth + 1
    with _PATCH_LOCK:
      for jc in discovered:
        jc_id = id(jc)
        existing = _JC_EXPORTED_PATCHES.get(jc_id)
        if existing is not None:
          _JC_EXPORTED_PATCHES[jc_id] = (
              existing[0],
              existing[1],
              existing[2] + 1,
          )
        else:
          orig_exported = jc.exported
          _JC_EXPORTED_PATCHES[jc_id] = (jc, orig_exported, 1)
          jc.exported = _make_eval_shape_fallback_exported(jc, orig_exported)
        installed.append(jc)
    yield
  finally:
    _THREAD_PATCH_STATE.eval_shape_fallback_depth = depth
    if installed:
      with _PATCH_LOCK:
        for jc in reversed(installed):
          jc_id = id(jc)
          entry = _JC_EXPORTED_PATCHES.get(jc_id)
          if entry is None:
            continue
          if entry[2] <= 1:
            jc.exported = entry[1]
            del _JC_EXPORTED_PATCHES[jc_id]
          else:
            _JC_EXPORTED_PATCHES[jc_id] = (entry[0], entry[1], entry[2] - 1)


def trace_to_fx_graph(
    fn_or_gm: Callable[..., Any],
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    tpu_version: findings.TpuVersion | None = None,
) -> fx.GraphModule:
  """Traces `fn_or_gm` into an `fx.GraphModule` on CPU using `FakeTensorMode`."""
  if isinstance(fn_or_gm, fx.GraphModule):
    return fn_or_gm
  kwargs = dict(example_kwargs or {})
  kw_keys = tuple(kwargs.keys())
  kw_vals = tuple(
      v.detach().clone() if isinstance(v, torch.Tensor) else v
      for v in kwargs.values()
  )
  num_pos = len(example_args)

  def _bound(*all_args):
    pos_args = all_args[:num_pos]
    call_kwargs = dict(zip(kw_keys, all_args[num_pos:]))
    return fn_or_gm(*pos_args, **call_kwargs)

  cloned_args = (
      tuple(
          a.detach().clone() if isinstance(a, torch.Tensor) else a
          for a in example_args
      )
      + kw_vals
  )
  # Module parameters and buffers are real tensors that the trace captures as
  # `get_attr` constants (`_allow_non_fake_inputs=True`).
  with (
      _preserve_module_state(fn_or_gm),
      _eval_shape_fallback_for_jax_callables((fn_or_gm, example_args, kwargs)),
  ):
    try:
      return proxy_tensor.make_fx(
          _bound, tracing_mode="fake", _allow_non_fake_inputs=True
      )(*cloned_args)
    except ValueError as exc:
      if (
          not _is_missing_tpu_device_kind_error(exc)
          or not jax.sharding.get_abstract_mesh().empty
      ):
        raise
      resolved_version = infer_tpu_version(fn_or_gm, tpu_version)
      with override_abstract_mesh(abstract_tpu_mesh(resolved_version)):
        return proxy_tensor.make_fx(
            _bound, tracing_mode="fake", _allow_non_fake_inputs=True
        )(*cloned_args)


def match_placeholders_to_inputs(
    placeholders: Sequence[fx.Node],
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
) -> dict[fx.Node, Any]:
  """Matches FX `placeholder` nodes to caller inputs in `trace_to_fx_graph` order."""
  top_level_inputs = list(example_args) + list((example_kwargs or {}).values())
  if len(placeholders) == len(top_level_inputs):
    return dict(zip(placeholders, top_level_inputs))
  flat_leaves = torch_pytree.tree_leaves(top_level_inputs)
  if len(placeholders) == len(flat_leaves):
    return dict(zip(placeholders, flat_leaves))
  flat_tensors = [
      leaf for leaf in flat_leaves if isinstance(leaf, torch.Tensor)
  ]
  if len(placeholders) == len(flat_tensors):
    return dict(zip(placeholders, flat_tensors))
  return {}


def _collect_gm_attr_storage_roots(
    gm: fx.GraphModule,
    ptr_to_root: dict[int, str],
) -> dict[fx.Node, set[str]]:
  """Assigns storage root IDs to `get_attr` tensor nodes across `gm`."""
  attr_roots: dict[fx.Node, set[str]] = {}
  for node in gm.graph.nodes:
    if node.op != "get_attr":
      continue
    attr_val = resolve_fx_attr(gm, str(node.target))
    if isinstance(attr_val, fx.GraphModule):
      continue
    root_name = f"attr:{node.target}"
    if isinstance(attr_val, torch.Tensor):
      ptr = _tensor_storage_ptr(attr_val)
      if ptr is not None:
        root_name = ptr_to_root.setdefault(ptr, root_name)
    attr_roots[node] = {root_name}
  return attr_roots


def _compute_placeholder_roots(
    placeholders: Sequence[fx.Node],
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None,
    *,
    ptr_to_root: dict[int, str] | None = None,
) -> dict[fx.Node, set[str]]:
  """Assigns storage root IDs to FX `placeholder` nodes.

  If two eager `torch.Tensor` arguments passed by the caller share underlying
  storage (or alias a module buffer/parameter in `ptr_to_root`), their FX
  placeholder nodes receive the same canonical root ID so cross-argument and
  input-to-buffer aliasing is tracked through the graph.

  Args:
    placeholders: FX `placeholder` nodes in graph order.
    example_args: Positional example arguments passed to `verify_fx_graph`.
    example_kwargs: Keyword example arguments passed to `verify_fx_graph`.
    ptr_to_root: Optional map from storage pointer to canonical root ID.

  Returns:
    Mapping from `placeholder` node to its single-element set of root IDs.
  """
  matched_map = match_placeholders_to_inputs(
      placeholders, example_args, example_kwargs
  )
  if not matched_map:
    return {ph: {ph.name} for ph in placeholders}

  storage_map = ptr_to_root if ptr_to_root is not None else {}
  placeholder_roots: dict[fx.Node, set[str]] = {}
  for ph_node, val in matched_map.items():
    matched_name = (
        f"param:{ph_node.name}"
        if isinstance(val, torch.nn.Parameter)
        else ph_node.name
    )
    if isinstance(val, torch.Tensor):
      ptr = _tensor_storage_ptr(val)
      if ptr is not None:
        matched_name = storage_map.setdefault(ptr, matched_name)
    placeholder_roots[ph_node] = {matched_name}
  return placeholder_roots


def _check_stale_root_uses(
    node: fx.Node,
    node_roots: Mapping[fx.Node, set[str]],
    invalidated_roots: Mapping[str, tuple[str, str]],
    *,
    jax_op_name: str | None = None,
    used_nodes_override: Sequence[fx.Node] | None = None,
    backward_pass: bool = False,
) -> list[findings.VerificationFinding]:
  """Checks whether `node` consumes any tensor derived from a donated buffer."""
  if jax_op_name is None and _is_metadata_only_fx_node(node):
    return []
  detected: list[findings.VerificationFinding] = []
  consumed = (
      list(used_nodes_override)
      if used_nodes_override is not None
      else iter_fx_nodes((node.args, node.kwargs))
  )
  for used_node in consumed:
    stale = node_roots.get(used_node, set()) & invalidated_roots.keys()
    for root in sorted(stale):
      donor_op, donor_node = invalidated_roots[root]
      if backward_pass:
        op_name = donor_op
        consumer_desc = (
            "autograd backward output"
            if node.op == "output"
            else f"autograd backward FX node '{node.name}' ({node.target})"
        )
        message = (
            f"{consumer_desc} reads '{used_node.name}' (sharing storage with"
            f" '{root}') after '{root}' was donated in forward to"
            f" '{donor_node}' ({donor_op})"
        )
      elif jax_op_name is not None:
        op_name = jax_op_name
        message = (
            f"FX node '{node.name}' ({jax_op_name}) reads '{used_node.name}'"
            f" derived from buffer '{root}', which was already donated and"
            f" invalidated by '{donor_node}' ({donor_op})"
        )
      else:
        op_name = donor_op
        consumer_desc = (
            "graph return (`output`)"
            if node.op == "output"
            else f"FX node '{node.name}' ({node.target})"
        )
        message = (
            f"{consumer_desc} uses '{used_node.name}' (sharing storage with"
            f" '{root}') after '{root}' was donated to '{donor_node}'"
            f" ({donor_op})"
        )
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.USE_AFTER_DONATE,
              severity=findings.Severity.ERROR,
              phase=_FX_GRAPH,
              op_name=op_name,
              source_location=node.name,
              message=message,
          )
      )
  return detected


def extract_hop_operands(
    node: fx.Node,
    sub_gm_nodes: Mapping[fx.Node, fx.GraphModule],
    num_placeholders: int,
) -> list[Any]:
  """Extracts the flat operand list passed to a higher-order op's subgraphs."""
  last_gm_pos = -1
  for idx, arg in enumerate(node.args):
    if isinstance(arg, fx.Node) and arg in sub_gm_nodes:
      last_gm_pos = idx
  trailing = node.args[last_gm_pos + 1 :]
  flattened: list[Any] = []
  for item in trailing:
    if isinstance(item, (tuple, list)):
      flattened.extend(item)
    else:
      flattened.append(item)
  if len(flattened) == num_placeholders:
    return flattened
  for item in reversed(trailing):
    if isinstance(item, (tuple, list)) and len(item) == num_placeholders:
      return list(item)
  return flattened


def _append_unique_hop_findings(
    items: Sequence[findings.VerificationFinding],
    seen_hop_keys: set[tuple[Any, ...]],
    detected: list[findings.VerificationFinding],
) -> None:
  """Appends findings from `items` whose dedup key is not in `seen_hop_keys`."""
  for item in items:
    key = (
        item.category,
        item.op_name,
        item.source_location,
        item.message,
    )
    if key not in seen_hop_keys:
      seen_hop_keys.add(key)
      detected.append(item)


def _analyze_fx_graph_module(
    gm: fx.GraphModule,
    placeholder_roots: Mapping[fx.Node, set[str]],
    initial_invalidated_roots: Mapping[str, tuple[str, str]],
    ptr_to_root: dict[int, str],
    *,
    backward_pass: bool = False,
    backward_boundary_op: Any = None,
) -> tuple[
    list[findings.VerificationFinding],
    dict[str, tuple[str, str]],
    list[tuple[set[str], bool]],
    bool,
]:
  """Runs storage-provenance dataflow over `gm` and any nested HOP subgraphs."""
  detected: list[findings.VerificationFinding] = []
  node_roots: dict[fx.Node, set[str]] = {}
  invalidated_roots: dict[str, tuple[str, str]] = dict(
      initial_invalidated_roots
  )
  fresh_nodes: set[fx.Node] = set()
  per_return_roots: dict[fx.Node, dict[int, set[str]]] = {}
  sub_gm_nodes: dict[fx.Node, fx.GraphModule] = {}
  attr_roots = _collect_gm_attr_storage_roots(gm, ptr_to_root)
  return_slots: list[tuple[set[str], bool]] = []
  has_invalidating_donor = False

  for node in gm.graph.nodes:
    if (
        backward_boundary_op is not None
        and node.op == "call_function"
        and node.target is backward_boundary_op
    ):
      backward_pass = True
      continue
    if node.op == "placeholder":
      node_roots[node] = set(placeholder_roots.get(node, {node.name}))
      continue

    if node.op == "get_attr":
      sub_val = resolve_fx_attr(gm, str(node.target))
      if isinstance(sub_val, fx.GraphModule):
        sub_gm_nodes[node] = sub_val
        continue
      node_roots[node] = set(attr_roots.get(node, {f"attr:{node.target}"}))
      continue

    if node.op == "call_function" and node.target is operator.getitem:
      container = node.args[0]
      index = node.args[1] if len(node.args) > 1 else None
      container_val = (
          container.meta.get("val") if isinstance(container, fx.Node) else None
      )
      if (
          isinstance(container, fx.Node)
          and container in per_return_roots
          and isinstance(index, int)
      ):
        slot_map = per_return_roots[container]
        if index in slot_map:
          node_roots[node] = set(slot_map[index])
        else:
          node_roots[node] = {f"val:{container.name}[{index}]"}
          fresh_nodes.add(node)
      elif (
          isinstance(container, fx.Node)
          and container in fresh_nodes
          and isinstance(index, int)
          and (
              container_val is None or isinstance(container_val, (tuple, list))
          )
      ):
        node_roots[node] = {f"{r}[{index}]" for r in node_roots[container]}
        fresh_nodes.add(node)
      elif isinstance(container, fx.Node):
        node_roots[node] = set(node_roots.get(container, {node.name}))
      else:
        node_roots[node] = {node.name}
      continue

    if _is_set_inplace_fx_node(node):
      detected.extend(
          _check_stale_root_uses(
              node,
              node_roots,
              invalidated_roots,
              used_nodes_override=iter_fx_nodes((node.args[1:], node.kwargs)),
              backward_pass=backward_pass,
          )
      )
      dst = node.args[0]
      if len(node.args) >= 2 and isinstance(node.args[1], fx.Node):
        src = node.args[1]
        new_roots = set(node_roots.get(src, {src.name}))
      else:
        new_roots = {f"set:{node.name}"}
      node_roots[dst] = set(new_roots)
      node_roots[node] = set(new_roots)
      if len(node.args) >= 2 and node.args[1] in fresh_nodes:
        fresh_nodes.add(dst)
        fresh_nodes.add(node)
      else:
        fresh_nodes.discard(dst)
      continue

    # Higher-order ops (`torch.cond`, `while_loop`, `scan`) holding child
    # `fx.GraphModule` attributes.
    child_gms = (
        [
            sub_gm_nodes[a]
            for a in node.args
            if isinstance(a, fx.Node) and a in sub_gm_nodes
        ]
        if node.op == "call_function"
        else []
    )
    if child_gms:
      detected.extend(
          _check_stale_root_uses(
              node,
              node_roots,
              invalidated_roots,
              backward_pass=backward_pass,
          )
      )
      merged_child_inv = dict(invalidated_roots)
      branch_ret_slots: list[list[tuple[set[str], bool]]] = []
      seen_hop_keys: set[tuple[Any, ...]] = set()

      target_str = str(node.target)
      is_while = "while_loop" in target_str
      is_scan = "scan" in target_str
      num_scan_carry: int | None = None
      if (
          is_scan
          and len(node.args) > 1
          and isinstance(node.args[1], (tuple, list))
      ):
        num_scan_carry = len(node.args[1])

      if is_while and len(child_gms) >= 2:
        cond_gm, body_gm = child_gms[0], child_gms[1]
        cond_phs = [n for n in cond_gm.graph.nodes if n.op == "placeholder"]
        cond_ops = extract_hop_operands(node, sub_gm_nodes, len(cond_phs))
        cond_ph_roots = {
            ph: (
                set(node_roots.get(op_val, {op_val.name}))
                if isinstance(op_val, fx.Node)
                else {ph.name}
            )
            for ph, op_val in zip(cond_phs, cond_ops)
        }
        c_findings, c_inv, _, c_donor = _analyze_fx_graph_module(
            cond_gm,
            cond_ph_roots,
            invalidated_roots,
            ptr_to_root,
            backward_pass=backward_pass,
        )
        has_invalidating_donor = has_invalidating_donor or c_donor
        _append_unique_hop_findings(c_findings, seen_hop_keys, detected)
        merged_child_inv.update(c_inv)

        body_phs = [n for n in body_gm.graph.nodes if n.op == "placeholder"]
        body_ops = extract_hop_operands(node, sub_gm_nodes, len(body_phs))
        body_ph_roots = {
            ph: (
                set(node_roots.get(op_val, {op_val.name}))
                if isinstance(op_val, fx.Node)
                else {ph.name}
            )
            for ph, op_val in zip(body_phs, body_ops)
        }
        # `while_loop` may exit on iteration 0 without running `body_gm`, so
        # initial carried_inputs can also flow directly to the output slots.
        num_while_carry = (
            len(node.args[2])
            if len(node.args) > 2 and isinstance(node.args[2], (tuple, list))
            else len(body_ops)
        )
        init_ret_slots = [
            (
                set(node_roots.get(op_val, {op_val.name}))
                if isinstance(op_val, fx.Node)
                else set(),
                isinstance(op_val, fx.Node) and op_val in fresh_nodes,
            )
            for op_val in body_ops[:num_while_carry]
        ]
        if init_ret_slots:
          branch_ret_slots.append(init_ret_slots)

        b_findings, b_inv, b_ret, b_donor = _analyze_fx_graph_module(
            body_gm,
            body_ph_roots,
            merged_child_inv,
            ptr_to_root,
            backward_pass=backward_pass,
        )
        has_invalidating_donor = has_invalidating_donor or b_donor
        _append_unique_hop_findings(b_findings, seen_hop_keys, detected)
        merged_child_inv.update(b_inv)
        if b_ret:
          branch_ret_slots.append(b_ret)

        if merged_child_inv != invalidated_roots:
          next_carry = b_ret
          for sub_gm_iter2 in (cond_gm, body_gm):
            phs2 = [
                n for n in sub_gm_iter2.graph.nodes if n.op == "placeholder"
            ]
            ops2 = extract_hop_operands(node, sub_gm_nodes, len(phs2))
            iter2_ph_roots = {
                ph: (
                    set(next_carry[idx][0])
                    if idx < len(next_carry)
                    else (
                        set(node_roots.get(op_val, {op_val.name}))
                        if isinstance(op_val, fx.Node)
                        else {ph.name}
                    )
                )
                for idx, (ph, op_val) in enumerate(zip(phs2, ops2))
            }
            c2_findings, c2_inv, _, _ = _analyze_fx_graph_module(
                sub_gm_iter2,
                iter2_ph_roots,
                merged_child_inv,
                ptr_to_root,
                backward_pass=backward_pass,
            )
            _append_unique_hop_findings(c2_findings, seen_hop_keys, detected)
            merged_child_inv.update(c2_inv)
      else:
        for child_gm in child_gms:
          child_phs = [n for n in child_gm.graph.nodes if n.op == "placeholder"]
          operands = extract_hop_operands(node, sub_gm_nodes, len(child_phs))
          child_ph_roots = {
              ph: (
                  set(node_roots.get(op_val, {op_val.name}))
                  if isinstance(op_val, fx.Node)
                  else {ph.name}
              )
              for ph, op_val in zip(child_phs, operands)
          }
          c_findings, c_inv, c_ret, c_donor = _analyze_fx_graph_module(
              child_gm,
              child_ph_roots,
              invalidated_roots,
              ptr_to_root,
              backward_pass=backward_pass,
          )
          has_invalidating_donor = has_invalidating_donor or c_donor
          _append_unique_hop_findings(c_findings, seen_hop_keys, detected)
          merged_child_inv.update(c_inv)
          if c_ret:
            if is_scan and num_scan_carry is not None:
              # `scan` returns `(*final_carry, *ys_stacked)`: stacked `ys`
              # outputs allocate fresh outer-dimension buffers, whereas carry
              # slots return `next_carry`.
              adjusted_ret = [
                  slot if idx < num_scan_carry else (set(), True)
                  for idx, slot in enumerate(c_ret)
              ]
              branch_ret_slots.append(adjusted_ret)
            else:
              branch_ret_slots.append(c_ret)

        if is_scan and merged_child_inv != invalidated_roots and child_gms:
          combine_gm = child_gms[0]
          raw_ret = branch_ret_slots[-1] if branch_ret_slots else []
          carry_limit = (
              num_scan_carry if num_scan_carry is not None else len(raw_ret)
          )
          child_phs = [
              n for n in combine_gm.graph.nodes if n.op == "placeholder"
          ]
          operands = extract_hop_operands(node, sub_gm_nodes, len(child_phs))
          iter2_ph_roots = {
              ph: (
                  set(raw_ret[idx][0])
                  if idx < carry_limit and idx < len(raw_ret)
                  else (
                      set(node_roots.get(op_val, {op_val.name}))
                      if isinstance(op_val, fx.Node)
                      else {ph.name}
                  )
              )
              for idx, (ph, op_val) in enumerate(zip(child_phs, operands))
          }
          c2_findings, c2_inv, _, _ = _analyze_fx_graph_module(
              combine_gm,
              iter2_ph_roots,
              merged_child_inv,
              ptr_to_root,
              backward_pass=backward_pass,
          )
          _append_unique_hop_findings(c2_findings, seen_hop_keys, detected)
          merged_child_inv.update(c2_inv)
      invalidated_roots.update(merged_child_inv)
      max_slots = max((len(s) for s in branch_ret_slots), default=0)
      if max_slots == 1:
        union_roots: set[str] = set()
        all_fresh = True
        for s in branch_ret_slots:
          if s:
            if s[0][1]:
              union_roots.add(f"out:{node.name}")
            else:
              union_roots.update(s[0][0])
              all_fresh = False
        node_roots[node] = union_roots or {f"out:{node.name}"}
        if all_fresh:
          fresh_nodes.add(node)
      elif max_slots > 1:
        slot_map: dict[int, set[str]] = {}
        union_all: set[str] = set()
        for slot_idx in range(max_slots):
          slot_roots: set[str] = set()
          slot_all_fresh = True
          for s in branch_ret_slots:
            if slot_idx < len(s):
              if s[slot_idx][1]:
                slot_roots.add(f"out:{node.name}[{slot_idx}]")
              else:
                slot_roots.update(s[slot_idx][0])
                slot_all_fresh = False
          if not slot_all_fresh and slot_roots:
            slot_map[slot_idx] = slot_roots
            union_all.update(slot_roots)
          else:
            union_all.add(f"out:{node.name}[{slot_idx}]")
        if slot_map:
          per_return_roots[node] = slot_map
          node_roots[node] = union_all
        else:
          node_roots[node] = {f"out:{node.name}"}
          fresh_nodes.add(node)
      else:
        node_roots[node] = {f"out:{node.name}"}
        fresh_nodes.add(node)
      continue

    jax_callable = (
        extract_jax_callable(node.target)
        if node.op == "call_function"
        else None
    )
    if jax_callable is not None:
      op_name = get_op_display_name(node.target)
      donated_set = _get_donated_arg_indices(
          jax_callable, len(node.args), include_io_aliases=True
      )
      invalidating_donated_set = _get_donated_arg_indices(
          jax_callable, len(node.args), include_io_aliases=False
      )
      if invalidating_donated_set:
        has_invalidating_donor = True

      # 1. Check whether any input to this `@jax_op` was already donated earlier.
      detected.extend(
          _check_stale_root_uses(
              node,
              node_roots,
              invalidated_roots,
              jax_op_name=op_name,
              backward_pass=backward_pass,
          )
      )

      # 2. Collect non-static argument FX nodes and check for aliased donors.
      arg_fx_nodes: list[tuple[int, fx.Node]] = []
      for arg_idx, arg_val in enumerate(node.args):
        if not is_static_positional_arg(jax_callable, arg_idx):
          for sub_node in iter_fx_nodes(arg_val):
            arg_fx_nodes.append((arg_idx, sub_node))
      # pylint: disable=protected-access
      for kw_name, kw_val in node.kwargs.items():
        if not _is_static_keyword_arg(jax_callable, kw_name):
          kw_idx = jax_callable._param_name_to_idx.get(kw_name, -1)
          for sub_node in iter_fx_nodes(kw_val):
            arg_fx_nodes.append((kw_idx, sub_node))
      # pylint: enable=protected-access

      if not backward_pass:
        for i_pos, (idx_a, n_a) in enumerate(arg_fx_nodes):
          roots_a = node_roots.get(n_a, set())
          for idx_b, n_b in arg_fx_nodes[i_pos + 1 :]:
            shared = roots_a & node_roots.get(n_b, set())
            if shared and (idx_a in donated_set or idx_b in donated_set):
              detected.append(
                  findings.VerificationFinding(
                      category=findings.FindingCategory.ALIASED_DONOR_INPUTS,
                      severity=findings.Severity.ERROR,
                      phase=_FX_GRAPH,
                      op_name=op_name,
                      source_location=node.name,
                      message=(
                          f"FX node '{node.name}' ({op_name}) receives aliased"
                          f" arguments '{n_a.name}' (arg {idx_a}) and"
                          f" '{n_b.name}' (arg {idx_b}) sharing root storage"
                          f" {sorted(shared)} while donate_argnums="
                          f"{tuple(sorted(donated_set))}"
                      ),
                  )
              )

      # 3. Invalidate provenance roots backing non-aliased donated arguments.
      # `input_output_aliases` copies the output back into the input tensor
      # rather than leaving its buffer dead (`pallas.py:1376-1377`).
      for arg_idx, sub_node in arg_fx_nodes:
        if arg_idx not in invalidating_donated_set:
          continue
        for root in sorted(node_roots.get(sub_node, set())):
          invalidated_roots[root] = (op_name, node.name)
          if not backward_pass and root.startswith(("attr:", "param:")):
            root_label = root.split(":", 1)[1]
            detected.append(
                findings.VerificationFinding(
                    category=findings.FindingCategory.DONATED_PARAMETER,
                    severity=findings.Severity.ERROR,
                    phase=_FX_GRAPH,
                    op_name=op_name,
                    source_location=node.name,
                    message=(
                        f"FX node '{node.name}' ({op_name}) donates arg"
                        f" {arg_idx} '{sub_node.name}', which is backed by"
                        f" module state or a parameter '{root_label}';"
                        " the kernel may reuse that buffer and corrupt it"
                        " for later calls, so donate a clone() instead"
                    ),
                )
            )

      # Output of `jax_op` is a newly materialized buffer (`mutates_args=()`).
      node_roots[node] = {f"out:{node.name}"}
      fresh_nodes.add(node)
      continue

    # Standard FX operation or `output` return node: check all consumed nodes.
    detected.extend(
        _check_stale_root_uses(
            node, node_roots, invalidated_roots, backward_pass=backward_pass
        )
    )

    if node.op == "output":
      raw_ret = node.args[0] if node.args else None
      ret_elems = (
          list(raw_ret) if isinstance(raw_ret, (tuple, list)) else [raw_ret]
      )
      for elem in ret_elems:
        if isinstance(elem, fx.Node):
          return_slots.append((
              set(node_roots.get(elem, {elem.name})),
              elem in fresh_nodes,
          ))
        else:
          return_slots.append((set(), True))
      continue

    if node.op == "call_function":
      per_ret = _fx_op_per_return_aliases(node.target, node.args, node.kwargs)
      if per_ret is not None:
        slot_roots_map: dict[int, set[str]] = {}
        merged_per_ret: set[str] = set()
        for ret_idx, src_nodes in per_ret.items():
          rset: set[str] = set()
          for src_node in src_nodes:
            rset.update(node_roots.get(src_node, {src_node.name}))
          if rset:
            slot_roots_map[ret_idx] = rset
            merged_per_ret.update(rset)
        if slot_roots_map:
          per_return_roots[node] = slot_roots_map
          node_roots[node] = merged_per_ret
          continue

      aliased_inputs = fx_op_aliases_inputs(node.target, node.args, node.kwargs)
      if aliased_inputs:
        merged_roots: set[str] = set()
        for src_node in aliased_inputs:
          merged_roots.update(node_roots.get(src_node, {src_node.name}))
        node_roots[node] = merged_roots
        continue
    elif node.op == "call_method":
      method_name = str(node.target)
      if method_name in _VIEW_FALLBACK_NAMES or (
          method_name.endswith("_") and not method_name.startswith("_")
      ):
        src = node.args[0] if node.args else None
        if isinstance(src, fx.Node):
          node_roots[node] = set(node_roots.get(src, {src.name}))
          continue

    node_roots[node] = {f"val:{node.name}"}
    fresh_nodes.add(node)

  return detected, invalidated_roots, return_slots, has_invalidating_donor


def _is_unregistered_autograd_error(exc: BaseException) -> bool:
  """Returns whether `exc` indicates an op with no registered backward."""
  msg = str(exc)
  return (
      "no autograd formula was registered" in msg
      or ("derivative for" in msg and "is not implemented" in msg)
      or "does not require grad and does not have a grad_fn" in msg
      or "element 0 of tensors does not require grad" in msg
  )


def _check_joint_autograd_donation(
    callable_ref: Callable[..., Any],
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None,
    forward_findings: Sequence[findings.VerificationFinding],
    *,
    tpu_version: findings.TpuVersion | None = None,
) -> list[findings.VerificationFinding]:
  """Traces forward + backward to catch saved activations donated in forward."""
  del forward_findings
  kwargs = dict(example_kwargs or {})
  root_mod = _extract_root_nn_module(callable_ref)
  has_grad_input = any(
      isinstance(leaf, torch.Tensor) and leaf.requires_grad
      for leaf in torch_pytree.tree_leaves((tuple(example_args), kwargs))
  )
  has_grad_param = root_mod is not None and any(
      p.requires_grad for p in root_mod.parameters()
  )
  if not has_grad_input and not has_grad_param:
    return []

  kw_keys = tuple(kwargs.keys())
  kw_vals = tuple(
      v.detach().clone().requires_grad_(v.requires_grad)
      if isinstance(v, torch.Tensor)
      else v
      for v in kwargs.values()
  )
  num_pos = len(example_args)
  cloned_args = (
      tuple(
          a.detach().clone().requires_grad_(a.requires_grad)
          if isinstance(a, torch.Tensor)
          else a
          for a in example_args
      )
      + kw_vals
  )

  ran_backward = False

  def _joint(*all_args: Any) -> Any:
    nonlocal ran_backward
    pos_args = all_args[:num_pos]
    call_kwargs = dict(zip(kw_keys, all_args[num_pos:]))
    with torch.enable_grad():
      out = callable_ref(*pos_args, **call_kwargs)
      diff_outs = [
          t
          for t in torch_pytree.tree_leaves(out)
          if isinstance(t, torch.Tensor)
          and t.requires_grad
          and (t.is_floating_point() or t.is_complex())
      ]
      grad_inputs = [
          t
          for t in torch_pytree.tree_leaves((pos_args, call_kwargs))
          if isinstance(t, torch.Tensor) and t.requires_grad
      ]
      if root_mod is not None:
        seen_ids = {id(t) for t in grad_inputs}
        for p in root_mod.parameters():
          if p.requires_grad and id(p) not in seen_ids:
            seen_ids.add(p_id := id(p))
            del p_id
            grad_inputs.append(p)
      if not diff_outs or not grad_inputs:
        return ()
      _ = _get_backward_boundary_op()(diff_outs[0].detach())
      cotangents = [torch.ones_like(t) for t in diff_outs]
      try:
        grads = torch.autograd.grad(
            diff_outs, grad_inputs, cotangents, allow_unused=True
        )
        ran_backward = True
        return grads
      except RuntimeError as exc:
        if not _is_unregistered_autograd_error(exc):
          raise
        collected_grads: list[torch.Tensor] = []
        for idx, out_t in enumerate(diff_outs):
          try:
            part_grads = torch.autograd.grad(
                [out_t],
                grad_inputs,
                [cotangents[idx]],
                retain_graph=True,
                allow_unused=True,
            )
          except RuntimeError as sub_exc:
            if _is_unregistered_autograd_error(sub_exc):
              continue
            raise
          ran_backward = True
          collected_grads.extend(g for g in part_grads if g is not None)
        return tuple(collected_grads)

  try:
    with (
        _preserve_module_state(callable_ref),
        _eval_shape_fallback_for_jax_callables(
            (callable_ref, example_args, kwargs)
        ),
    ):
      try:
        joint_gm = proxy_tensor.make_fx(
            _joint, tracing_mode="fake", _allow_non_fake_inputs=True
        )(*cloned_args)
      except ValueError as exc:
        if (
            not _is_missing_tpu_device_kind_error(exc)
            or not jax.sharding.get_abstract_mesh().empty
        ):
          raise
        resolved_version = infer_tpu_version(callable_ref, tpu_version)
        with override_abstract_mesh(abstract_tpu_mesh(resolved_version)):
          joint_gm = proxy_tensor.make_fx(
              _joint, tracing_mode="fake", _allow_non_fake_inputs=True
          )(*cloned_args)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if _is_unregistered_autograd_error(exc):
      return []
    return [
        findings.VerificationFinding(
            category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
            severity=findings.Severity.WARNING,
            phase=_FX_GRAPH,
            op_name=get_op_display_name(callable_ref),
            message=(
                "could not trace joint forward+backward autograd graph for"
                f" buffer donation checks ({type(exc).__name__}: {exc})"
            ),
        )
    ]

  if not ran_backward:
    return []

  joint_ptr_to_root: dict[int, str] = {}
  _collect_gm_attr_storage_roots(joint_gm, joint_ptr_to_root)
  joint_phs = [n for n in joint_gm.graph.nodes if n.op == "placeholder"]
  joint_ph_roots = _compute_placeholder_roots(
      joint_phs, example_args, kwargs, ptr_to_root=joint_ptr_to_root
  )
  joint_findings, _, _, _ = _analyze_fx_graph_module(
      joint_gm,
      joint_ph_roots,
      {},
      joint_ptr_to_root,
      backward_boundary_op=_get_backward_boundary_op(),
  )
  return [
      f
      for f in joint_findings
      if f.category == findings.FindingCategory.USE_AFTER_DONATE
      and "autograd backward" in f.message
  ]


def verify_fx_graph(
    fn_or_gm: Callable[..., Any],
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    tpu_version: findings.TpuVersion | None = None,
    target_callable: Callable[..., Any] | None = None,
) -> findings.VerificationReport:
  """Verifies linear buffer donation and alias invariants in a PyTorch FX graph.

  Because `@jax_op` registers `mutates_args=()` on its PyTorch custom op (so
  Dynamo does not copy back donated inputs), PyTorch's functionalization pass
  does not track buffer invalidation from `donate_argnums`. This pass performs
  a forward storage-provenance analysis across the traced FX graph (including
  nested higher-order subgraphs and, when differentiable inputs or parameters
  are present, the joint forward+backward autograd graph):

  * Every FX node is mapped to the set of underlying storage root IDs it may
    alias (`node_roots`), unifying aliased caller inputs and module buffers.
  * View operations (`view`, `reshape`, `slice`, `.T`, `.mT`, tensor slicing,
    and in-place ops) propagate the storage roots of their input tensor(s),
    while `Tensor.set_` rebinds the destination node to its new source roots.
  * Tuple unpacking (`operator.getitem` on a multi-output op in `fresh_nodes`
    or per-return `out=(a, b)` ops) tracks independent storage roots per slot.
  * When a `@jax_op` donates an argument via `donate_argnums`, all storage roots
    backing that argument are added to `invalidated_roots`. Any subsequent read,
    in-place copy (`pool.copy_(result)`), backward read of a saved activation,
    or graph return of a node sharing an invalidated root is flagged as
    `USE_AFTER_DONATE`.

  Args:
    fn_or_gm: A PyTorch callable, `nn.Module`, or pre-traced `fx.GraphModule`.
    example_args: Example positional arguments.
    example_kwargs: Example keyword arguments.
    tpu_version: Target TPU generation used if tracing lowers an op.
    target_callable: Optional original callable when `fn_or_gm` is already a
      traced `fx.GraphModule`, enabling joint forward+backward tracing.

  Returns:
    A `VerificationReport` for `Phase.FX_GRAPH`.
  """
  gm = trace_to_fx_graph(
      fn_or_gm, example_args, example_kwargs, tpu_version=tpu_version
  )
  ptr_to_root: dict[int, str] = {}
  _collect_gm_attr_storage_roots(gm, ptr_to_root)
  placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
  placeholder_roots = _compute_placeholder_roots(
      placeholders, example_args, example_kwargs, ptr_to_root=ptr_to_root
  )
  detected, _, _, has_invalidating_donor = _analyze_fx_graph_module(
      gm, placeholder_roots, {}, ptr_to_root
  )

  callable_ref = target_callable or (
      None if isinstance(fn_or_gm, fx.GraphModule) else fn_or_gm
  )
  if has_invalidating_donor and callable_ref is not None:
    detected.extend(
        _check_joint_autograd_donation(
            callable_ref,
            example_args,
            example_kwargs,
            detected,
            tpu_version=tpu_version,
        )
    )

  return findings.report_of(_FX_GRAPH, detected)


# ==============================================================================
# 6. CPU TPU-Interpret Execution & Custom Autograd Parity Verification
# ==============================================================================


def tensor_to_numpy(tensor: torch.Tensor) -> tuple[np.ndarray, Any]:
  """Returns `(host_numpy_array, target_jax_dtype)` for a `torch.Tensor`.

  Args:
    tensor: Input `torch.Tensor` (including `bfloat16` and `float8_*` dtypes).

  Returns:
    A `(numpy_array, jax_dtype)` pair ready to pass to `jnp.asarray`.
  """
  detached = tensor.detach().cpu()
  jax_dtype = pallas_lib.TORCH_TO_JAX_DTYPE_MAP.get(detached.dtype)
  if jax_dtype is None:
    raise NotImplementedError(
        f"unsupported dtype for interpret: {detached.dtype}"
    )
  if detached.dtype == torch.bfloat16:
    return detached.to(torch.float32).numpy(), jnp.bfloat16
  if detached.dtype in FLOAT8_TORCH_DTYPES:
    return (
        detached.contiguous().view(torch.uint8).numpy().view(jax_dtype),
        jax_dtype,
    )
  return detached.numpy(), jax_dtype


def build_sharded_cpu_array(
    local_tensor: torch.Tensor,
    cpu_mesh: jax.sharding.Mesh,
    spec: Any,
    *,
    distinguish_ranks: bool = False,
) -> jax.Array:
  """Builds a global sharded `jax.Array` on `cpu_mesh` from a local shard."""
  base_np, jax_dtype = tensor_to_numpy(local_tensor)
  sharding = jax.sharding.NamedSharding(
      cpu_mesh,
      spec if spec is not None else jax.sharding.PartitionSpec(),
  )
  global_shape = pallas_lib.get_global_shape(
      tuple(local_tensor.shape), cpu_mesh, spec
  )
  use_rank_offset = distinguish_ranks and global_shape != tuple(
      local_tensor.shape
  )

  per_device_arrays: list[jax.Array] = []
  for flat_rank, dev in enumerate(cpu_mesh.devices.flat):
    if use_rank_offset and np.issubdtype(base_np.dtype, np.floating):
      shard_np = base_np + np.asarray(flat_rank * 0.25, dtype=base_np.dtype)
    elif (
        use_rank_offset
        and jax_dtype in FLOAT8_JAX_DTYPES
        and jax_dtype != jnp.float8_e8m0fnu.dtype
    ):
      shard_np = (
          base_np.astype(np.float32) + np.float32(flat_rank * 0.25)
      ).astype(base_np.dtype)
    elif (
        use_rank_offset
        and np.issubdtype(base_np.dtype, np.integer)
        and base_np.dtype != np.bool_
    ):
      shard_np = base_np + np.asarray(flat_rank, dtype=base_np.dtype)
    else:
      shard_np = base_np
    per_device_arrays.append(
        jax.device_put(jnp.asarray(shard_np, dtype=jax_dtype), dev)
    )
  return jax.make_array_from_single_device_arrays(
      global_shape, sharding, per_device_arrays
  )


def _torch_to_jax_array(tensor: torch.Tensor, device: jax.Device) -> jax.Array:
  """Converts a CPU `torch.Tensor` into a `jax.Array` on `device`."""
  np_val, jax_dtype = tensor_to_numpy(tensor)
  return jax.device_put(jnp.asarray(np_val, dtype=jax_dtype), device)


def _jax_to_torch_tensor(
    arr: jax.Array, *, local_shard: bool = False
) -> torch.Tensor:
  """Converts a CPU `jax.Array` (or its rank-0 shard) into a `torch.Tensor`."""
  target = (
      arr.addressable_shards[0].data
      if local_shard and arr.addressable_shards
      else arr
  )
  np_arr = np.asarray(target)
  if target.dtype == jnp.bfloat16:
    return torch.from_numpy(np_arr.astype(np.float32)).to(torch.bfloat16)
  if target.dtype in FLOAT8_JAX_DTYPES:
    torch_dtype = pallas_lib.JAX_TO_TORCH_DTYPE_MAP[target.dtype]
    return torch.from_numpy(
        np.asarray(np_arr, order="C").view(np.uint8).copy()
    ).view(torch_dtype)
  return torch.from_numpy(np_arr.copy())


def _run_jax_callable_on_cpu_interpret(
    jax_callable: pallas_lib.JaxCallable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Any:
  """Executes a single-chip or multi-chip `JaxCallable` on CPU interpret mode."""
  raw_fn, mesh_or_cpu_mesh = unwrap_raw_jax_fn(
      jax_callable, prefer_cpu_mesh=True
  )
  if jax_callable.mesh is not None and is_mock_tpu_mesh(mesh_or_cpu_mesh):
    raise ValueError(
        f"CPU interpret requires at least {jax_callable.mesh.size} virtual CPU"
        f" devices, got {len(jax.devices('cpu'))}"
    )
  cpu_mesh: jax.sharding.Mesh | None = mesh_or_cpu_mesh  # type: ignore[assignment]
  # pylint: disable=protected-access
  norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
      args, dict(kwargs)
  )
  # pylint: enable=protected-access
  norm_args = replace_mock_meshes_in_pytree(norm_args, cpu_mesh)
  norm_kwargs = replace_mock_meshes_in_pytree(norm_kwargs, cpu_mesh)
  cpu0 = jax.devices("cpu")[0]
  in_specs = jax_callable.input_partition_specs

  def _to_jax_input(val: Any, spec: Any = None) -> Any:
    if isinstance(val, torch.Tensor):
      if cpu_mesh is not None:
        return build_sharded_cpu_array(
            val, cpu_mesh, spec, distinguish_ranks=True
        )
      return _torch_to_jax_array(val, cpu0)
    if isinstance(val, (list, tuple)):
      return type(val)(_to_jax_input(elem, spec) for elem in val)
    return val

  with cpu_interpret_mesh_context(cpu_mesh):
    jax_args = [
        a
        if is_static_positional_arg(jax_callable, i)
        else _to_jax_input(
            a,
            in_specs[i] if in_specs is not None and i < len(in_specs) else None,
        )
        for i, a in enumerate(norm_args)
    ]
    jax_kwargs = {k: _to_jax_input(v) for k, v in norm_kwargs.items()}

    params = pltpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=False,
        uninitialized_memory="zero",
    )
    dev_ctx = (
        contextlib.nullcontext()
        if cpu_mesh is not None
        else jax.default_device(cpu0)
    )
    try:
      with dev_ctx, pltpu.force_tpu_interpret_mode(params):
        out = raw_fn(*jax_args, **jax_kwargs)
        jax.effects_barrier()
    finally:
      pltpu.reset_tpu_interpret_mode_state()

    return jax.tree.map(
        lambda leaf: _jax_to_torch_tensor(
            leaf, local_shard=(cpu_mesh is not None)
        )
        if isinstance(leaf, jax.Array)
        else leaf,
        out,
    )


class _CpuPallasInterpretDispatchMode(_python_dispatch.TorchDispatchMode):
  """Intercepts `@pallas.jax_op` calls below Autograd and runs them on CPU."""

  def __torch_dispatch__(self, func, types_unused, args=(), kwargs=None):
    del types_unused
    kwargs = kwargs or {}
    jax_callable = extract_jax_callable(func)
    if jax_callable is not None:
      return _run_jax_callable_on_cpu_interpret(jax_callable, args, kwargs)
    return func(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class _DiffInputSlot:
  """Tracks one differentiable tensor input to a `@jax_op` during autograd checks.

  Attributes:
    is_kwarg: `True` if passed in `example_kwargs`, `False` if in
      `example_args`.
    key: Positional index (`int`) or keyword name (`str`).
    param_idx: Parameter index in `JaxCallable._orig_params` (for diagnostics).
    tensor: Cloned CPU `torch.Tensor` with `requires_grad=True`.
  """

  is_kwarg: bool
  key: int | str
  param_idx: int
  tensor: torch.Tensor


def _collect_differentiable_inputs(
    jax_callable: pallas_lib.JaxCallable,
    example_args: Sequence[Any],
    raw_kwargs: Mapping[str, Any],
) -> tuple[list[_DiffInputSlot], list[Any], dict[str, Any], bool]:
  """Collects differentiable input slots without expanding `functools.partial` defaults."""
  diff_slots: list[_DiffInputSlot] = []
  pt_args: list[Any] = []
  pt_kwargs: dict[str, Any] = {}
  has_complex = False

  def _is_diff_candidate(val: Any) -> bool:
    return (
        isinstance(val, torch.Tensor)
        and val.dtype not in FLOAT8_TORCH_DTYPES
        and (val.is_floating_point() or val.is_complex())
    )

  for idx, arg in enumerate(example_args):
    if not is_static_positional_arg(jax_callable, idx) and _is_diff_candidate(
        arg
    ):
      has_complex = has_complex or arg.is_complex()
      cloned = arg.detach().cpu().clone().requires_grad_(True)
      pt_args.append(cloned)
      diff_slots.append(
          _DiffInputSlot(is_kwarg=False, key=idx, param_idx=idx, tensor=cloned)
      )
    else:
      pt_args.append(arg)

  # pylint: disable=protected-access
  for k, arg in raw_kwargs.items():
    if not _is_static_keyword_arg(jax_callable, k) and _is_diff_candidate(arg):
      has_complex = has_complex or arg.is_complex()
      cloned = arg.detach().cpu().clone().requires_grad_(True)
      pt_kwargs[k] = cloned
      param_idx = jax_callable._param_name_to_idx.get(k, -1)
      diff_slots.append(
          _DiffInputSlot(
              is_kwarg=True,
              key=k,
              param_idx=param_idx if param_idx >= 0 else len(pt_args),
              tensor=cloned,
          )
      )
    else:
      pt_kwargs[k] = arg
  # pylint: enable=protected-access
  return diff_slots, pt_args, pt_kwargs, has_complex


def _build_normalized_cotangents(
    tensor_outs: Sequence[torch.Tensor],
    differentiable: Sequence[bool],
    rng: torch.Generator,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
  """Generates deterministic unit-L2-norm cotangents for `tensor_outs`."""
  raw_cotangents = [
      torch.randn(t.shape, dtype=torch.float32, generator=rng)
      if is_diff
      else torch.zeros(t.shape, dtype=torch.float32)
      for t, is_diff in zip(tensor_outs, differentiable)
  ]
  cot_norm = (
      float(sum(float((g * g).sum().item()) for g in raw_cotangents) ** 0.5)
      or 1.0
  )
  cotangents_pt = [
      (g / cot_norm).to(t.dtype)
      if is_diff
      else (
          torch.zeros(t.shape, dtype=torch.uint8).view(t.dtype)
          if t.dtype in FLOAT8_TORCH_DTYPES
          else torch.zeros(t.shape, dtype=t.dtype)
      )
      for g, t, is_diff in zip(raw_cotangents, tensor_outs, differentiable)
  ]
  cotangents_f32 = [
      c.to(torch.float32) if is_diff else torch.zeros_like(g)
      for c, g, is_diff in zip(cotangents_pt, raw_cotangents, differentiable)
  ]
  return cotangents_pt, cotangents_f32


def _try_analytical_jax_vjp(
    jax_callable: pallas_lib.JaxCallable,
    diff_slots: Sequence[_DiffInputSlot],
    pt_args: Sequence[Any],
    pt_kwargs: Mapping[str, Any],
    cotangents_pt: Sequence[torch.Tensor],
) -> Sequence[Any] | None:
  """Computes `jax.vjp` gradients on CPU interpret mode if supported by the forward op."""
  raw_fwd_fn, cpu_mesh = unwrap_raw_jax_fn(jax_callable, prefer_cpu_mesh=True)
  if cpu_mesh is not None:
    return None

  cpu0 = jax.devices("cpu")[0]
  with cpu_interpret_mesh_context(None):
    jax_primals = [
        _torch_to_jax_array(slot.tensor, cpu0) for slot in diff_slots
    ]

    def _pure_diff_fwd(*diff_jax_arrays: Any) -> Any:
      call_pt_args = list(pt_args)
      call_pt_kwargs = dict(pt_kwargs)
      for slot, arr in zip(diff_slots, diff_jax_arrays):
        if slot.is_kwarg:
          call_pt_kwargs[str(slot.key)] = arr
        else:
          call_pt_args[int(slot.key)] = arr
      # pylint: disable=protected-access
      norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
          call_pt_args, call_pt_kwargs
      )
      # pylint: enable=protected-access
      norm_args = replace_mock_meshes_in_pytree(norm_args, cpu_mesh)
      norm_kwargs = replace_mock_meshes_in_pytree(norm_kwargs, cpu_mesh)
      call_args = jax.tree.map(
          lambda v: _torch_to_jax_array(v, cpu0)
          if isinstance(v, torch.Tensor)
          else v,
          norm_args,
      )
      call_kwargs = jax.tree.map(
          lambda v: _torch_to_jax_array(v, cpu0)
          if isinstance(v, torch.Tensor)
          else v,
          norm_kwargs,
      )
      return jax.tree.map(
          lambda leaf: jax.lax.stop_gradient(leaf)
          if isinstance(leaf, jax.Array) and leaf.dtype in FLOAT8_JAX_DTYPES
          else leaf,
          raw_fwd_fn(*call_args, **call_kwargs),
      )

    params = pltpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=False,
        uninitialized_memory="zero",
    )
    try:
      with jax.default_device(cpu0), pltpu.force_tpu_interpret_mode(params):
        jax_fwd_out, vjp_fn = jax.vjp(_pure_diff_fwd, *jax_primals)
        out_leaves, out_tree = jax.tree.flatten(jax_fwd_out)
        cot_leaves = []
        for out_leaf, g in zip(out_leaves, cotangents_pt):
          if (
              hasattr(out_leaf, "dtype")
              and not jnp.issubdtype(out_leaf.dtype, jnp.floating)
              and not jnp.issubdtype(out_leaf.dtype, jnp.complexfloating)
              and out_leaf.dtype not in FLOAT8_JAX_DTYPES
          ):
            cot_leaves.append(np.zeros(g.shape, dtype=jax.dtypes.float0))
          else:
            cot_leaves.append(_torch_to_jax_array(g, cpu0))
        jax_cotangents = out_tree.unflatten(cot_leaves)
        ref_jax_grads = vjp_fn(jax_cotangents)
        jax.effects_barrier()
        return ref_jax_grads
    except (
        ValueError,
        NotImplementedError,
        TypeError,
        AssertionError,
        RuntimeError,
    ):
      return None
    finally:
      pltpu.reset_tpu_interpret_mode_state()


def _compare_vjp_gradients(
    diff_slots: Sequence[_DiffInputSlot],
    pt_grads: Sequence[torch.Tensor | None],
    ref_jax_grads: Sequence[Any],
    tensor_outs: Sequence[torch.Tensor],
    differentiable: Sequence[bool],
    op_name: str,
    *,
    rtol: float,
    atol: float,
) -> list[findings.VerificationFinding]:
  """Compares custom PyTorch backward gradients against analytical `jax.vjp`."""
  detected: list[findings.VerificationFinding] = []
  has_low_prec_out = any(
      t.dtype in (torch.bfloat16, torch.float16)
      for t, d in zip(tensor_outs, differentiable)
      if d
  )
  for slot, pt_grad, ref_jax_grad in zip(diff_slots, pt_grads, ref_jax_grads):
    ref_pt_grad = _jax_to_torch_tensor(ref_jax_grad)
    is_low_prec = has_low_prec_out or slot.tensor.dtype in (
        torch.bfloat16,
        torch.float16,
    )
    vjp_rtol = max(rtol, 2e-2 if is_low_prec else rtol)
    vjp_atol = max(atol, 2e-2 if is_low_prec else atol)
    if pt_grad is None:
      ref_max = float(ref_pt_grad.abs().max().item())
      if ref_max > vjp_atol:
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_AUTOGRAD,
                op_name=op_name,
                message=(
                    "custom backward returned None for differentiable input"
                    f" arg {slot.param_idx}, whereas jax.vjp reference has max"
                    f" magnitude {ref_max:.4e}"
                ),
            )
        )
      continue
    if not torch.allclose(
        pt_grad.to(torch.float32),
        ref_pt_grad.to(torch.float32),
        rtol=vjp_rtol,
        atol=vjp_atol,
    ):
      max_diff = float(
          (pt_grad.to(torch.float32) - ref_pt_grad.to(torch.float32))
          .abs()
          .max()
          .item()
      )
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_AUTOGRAD,
              op_name=op_name,
              message=(
                  f"custom PyTorch backward for arg {slot.param_idx} diverges"
                  " from jax.vjp of forward Pallas kernel (max_abs_diff="
                  f"{max_diff:.4e}, rtol={vjp_rtol}, atol={vjp_atol})"
              ),
          )
      )
  return detected


def _compare_finite_difference_gradients(
    jax_callable: pallas_lib.JaxCallable,
    diff_slots: Sequence[_DiffInputSlot],
    pt_grads: Sequence[torch.Tensor | None],
    pt_args: Sequence[Any],
    pt_kwargs: Mapping[str, Any],
    cotangents_f32: Sequence[torch.Tensor],
    differentiable: Sequence[bool],
    rng: torch.Generator,
    op_name: str,
    *,
    rtol: float,
    atol: float,
) -> list[findings.VerificationFinding]:
  """Compares custom backward gradients against central finite differences.

  Bare `pl.pallas_call` and multi-chip `jax.shard_map` primitives lack default
  JAX JVP rules. For each differentiable input `x`, we probe the directional
  derivative along:
  1. The normalized custom backward gradient direction (which immediately
     exposes wrong scale factors, sign errors, or transposed dimensions),
  2. 4 random Rademacher (`+/-1`) directions, and
  3. Up to 2 localized single-coordinate (`one-hot`) boundary directions (first
     and last elements) so single-row or single-tile errors are not diluted by
     `sqrt(numel)` averaging.

  Perturbations `x_pos` and `x_neg` are rounded to `x.dtype` before evaluating
  the forward kernel, and the directional projection uses the exact quantized
  step `delta_x = (x_pos - x_neg) / 2` so low-precision (`bfloat16`/`float16`)
  grid snapping is accounted for.
  """
  detected: list[findings.VerificationFinding] = []
  base_detached_args = [
      a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in pt_args
  ]
  base_detached_kwargs = {
      k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
      for k, v in pt_kwargs.items()
  }

  for slot, pt_grad in zip(diff_slots, pt_grads):
    x_base = slot.tensor.detach().cpu()
    orig_dtype = x_base.dtype
    x_f32 = x_base.to(torch.float32)
    is_low_prec = orig_dtype in (torch.float16, torch.bfloat16)
    eps = 6.25e-2 if is_low_prec else (2.0**-10)
    rel_eps = (
        2.0**-4
        if orig_dtype == torch.bfloat16
        else (2.0**-6 if orig_dtype == torch.float16 else 2.0**-10)
    )
    step = torch.maximum(torch.full_like(x_f32, eps), rel_eps * x_f32.abs())
    fd_atol = max(atol, 5e-2 if is_low_prec else 2e-3)
    fd_rtol = max(rtol, 5e-2 if is_low_prec else 5e-3)

    probes: list[torch.Tensor] = []
    if pt_grad is not None:
      grad_f32 = pt_grad.detach().cpu().to(torch.float32)
      grad_max = float(grad_f32.abs().max().item())
      if grad_max > 1e-6:
        probes.append(grad_f32 / grad_max)
    for _ in range(4):
      bits = torch.randint(0, 2, x_base.shape, generator=rng, dtype=torch.int64)
      probes.append(bits.to(torch.float32) * 2.0 - 1.0)
    numel = x_base.numel()
    for flat_idx in (0, numel - 1) if numel > 1 else (0,) if numel == 1 else ():
      one_hot = torch.zeros_like(x_f32)
      one_hot.view(-1)[flat_idx] = 1.0
      probes.append(one_hot)

    for probe_idx, v_f32 in enumerate(probes):
      x_pos = (x_f32 + step * v_f32).to(orig_dtype)
      x_neg = (x_f32 - step * v_f32).to(orig_dtype)
      delta_x = (x_pos.to(torch.float32) - x_neg.to(torch.float32)) * 0.5
      delta_norm = math.sqrt(float(delta_x.pow(2).sum().item()))
      if delta_norm == 0.0:
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
                severity=findings.Severity.WARNING,
                phase=_AUTOGRAD,
                op_name=op_name,
                message=(
                    f"finite-difference perturbation on arg {slot.param_idx}"
                    f" rounded to zero in {orig_dtype}"
                ),
            )
        )
        break

      args_pos = list(base_detached_args)
      args_neg = list(base_detached_args)
      kwargs_pos = dict(base_detached_kwargs)
      kwargs_neg = dict(base_detached_kwargs)
      if slot.is_kwarg:
        kwargs_pos[str(slot.key)] = x_pos
        kwargs_neg[str(slot.key)] = x_neg
      else:
        args_pos[int(slot.key)] = x_pos
        args_neg[int(slot.key)] = x_neg

      try:
        out_pos = _run_jax_callable_on_cpu_interpret(
            jax_callable, args_pos, kwargs_pos
        )
        out_neg = _run_jax_callable_on_cpu_interpret(
            jax_callable, args_neg, kwargs_neg
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
                severity=findings.Severity.WARNING,
                phase=_AUTOGRAD,
                op_name=op_name,
                message=(
                    "skipped finite-difference autograd check because CPU"
                    f" interpret raised {type(exc).__name__}: {exc}"
                ),
            )
        )
        return detected

      pos_tensors = [
          t for t in jax.tree.leaves(out_pos) if isinstance(t, torch.Tensor)
      ]
      neg_tensors = [
          t for t in jax.tree.leaves(out_neg) if isinstance(t, torch.Tensor)
      ]

      phi_diff = sum(
          float((u * (p.to(torch.float32) - n.to(torch.float32))).sum().item())
          for u, p, n, d in zip(
              cotangents_f32, pos_tensors, neg_tensors, differentiable
          )
          if d
      )
      fd_dir = phi_diff / (2.0 * delta_norm)
      pt_dir = (
          0.0
          if pt_grad is None
          else float(
              (pt_grad.detach().cpu().to(torch.float32) * delta_x).sum().item()
          )
          / delta_norm
      )
      abs_err = abs(pt_dir - fd_dir)
      if abs_err > fd_atol + fd_rtol * abs(fd_dir):
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_AUTOGRAD,
                op_name=op_name,
                message=(
                    f"custom PyTorch backward for arg {slot.param_idx} diverges"
                    " from central finite-difference directional JVP of"
                    f" forward Pallas kernel on probe #{probe_idx}"
                    f" (backward_dir={pt_dir:.4e}, fd_ref_dir={fd_dir:.4e},"
                    f" abs_err={abs_err:.4e})"
                ),
            )
        )
        break
  return detected


def _check_saved_donated_inputs(
    jax_callable: pallas_lib.JaxCallable,
    pt_args: Sequence[Any],
    pt_kwargs: Mapping[str, Any],
    saved_tensors: Sequence[torch.Tensor],
    op_name: str,
) -> list[findings.VerificationFinding]:
  """Checks whether `setup_context` saved any tensor sharing storage with a donated input."""
  if not saved_tensors:
    return []
  # pylint: disable=protected-access
  norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
      pt_args, dict(pt_kwargs)
  )
  # pylint: enable=protected-access
  donated_set = _get_donated_arg_indices(jax_callable, len(norm_args))
  if not donated_set:
    return []

  tensor_entries = _collect_jax_callable_tensor_args(
      jax_callable, norm_args, norm_kwargs
  )
  detected: list[findings.VerificationFinding] = []
  for param_idx, name, donated_tensor in tensor_entries:
    if param_idx not in donated_set:
      continue
    if any(
        _eager_tensors_alias(saved, donated_tensor) for saved in saved_tensors
    ):
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.USE_AFTER_DONATE,
              severity=findings.Severity.ERROR,
              phase=_AUTOGRAD,
              op_name=op_name,
              message=(
                  f"setup_context saves donated input {name!r} (arg"
                  f" {param_idx}) for backward while"
                  f" donate_argnums={tuple(sorted(donated_set))}; on TPU the"
                  " forward kernel reuses or invalidates donated input"
                  " buffers before backward runs"
              ),
          )
      )
  return detected


def verify_autograd_parity(
    op_def: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> findings.VerificationReport:
  """Checks a registered PyTorch custom autograd backward against the forward op."""
  resolved = resolve_custom_op_def(op_def)
  if resolved is None or resolved._backward_fn is None:  # pylint: disable=protected-access
    return findings.VerificationReport()

  jax_callable = extract_jax_callable(resolved)
  if jax_callable is None:
    return findings.VerificationReport()

  op_name = resolved._qualname  # pylint: disable=protected-access
  diff_slots, pt_args, pt_kwargs, has_complex = _collect_differentiable_inputs(
      jax_callable, example_args, dict(example_kwargs or {})
  )
  if has_complex:
    return findings.report_of(
        _AUTOGRAD,
        [
            findings.VerificationFinding(
                category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
                severity=findings.Severity.INFO,
                phase=_AUTOGRAD,
                op_name=op_name,
                message=(
                    "skipped autograd parity check because complex-valued"
                    " differentiable inputs are not supported"
                ),
            )
        ],
    )
  if not diff_slots:
    return findings.VerificationReport()

  # 1. Run the PyTorch forward pass under CPU TPU-interpret dispatch and record
  # any tensors saved by `setup_context` for backward.
  saved_for_backward: list[torch.Tensor] = []
  # pylint: disable=protected-access
  orig_setup_context: Callable[..., Any] | None = getattr(
      resolved, "_setup_context_fn", None
  )

  def _wrapped_setup_context(*s_args: Any, **s_kwargs: Any) -> Any:
    res = (
        orig_setup_context(*s_args, **s_kwargs)
        if callable(orig_setup_context)
        else None
    )
    ctx = (
        s_kwargs.get("ctx")
        if "ctx" in s_kwargs
        else (s_args[0] if s_args else None)
    )
    if ctx is not None:
      for t in getattr(ctx, "to_save", None) or ():
        if isinstance(t, torch.Tensor):
          saved_for_backward.append(t)
      for v in getattr(ctx, "__dict__", {}).values():
        for leaf in torch_pytree.tree_leaves(v):
          if isinstance(leaf, torch.Tensor):
            saved_for_backward.append(leaf)
    return res

  try:
    if orig_setup_context is not None:
      resolved._setup_context_fn = _wrapped_setup_context
    with _CpuPallasInterpretDispatchMode():
      pt_out = resolved(*pt_args, **pt_kwargs)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return findings.report_of(
        _AUTOGRAD,
        [
            findings.VerificationFinding(
                category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
                severity=findings.Severity.INFO,
                phase=_AUTOGRAD,
                op_name=op_name,
                message=f"skipped autograd parity check ({exc})",
            )
        ],
    )
  finally:
    if orig_setup_context is not None:
      resolved._setup_context_fn = orig_setup_context
  # pylint: enable=protected-access

  saved_donor_findings = _check_saved_donated_inputs(
      jax_callable, pt_args, pt_kwargs, saved_for_backward, op_name
  )

  tensor_outs = [
      t for t in jax.tree.leaves(pt_out) if isinstance(t, torch.Tensor)
  ]
  # Integer, quantized FP8/E8M0, and non-dependent outputs have no continuous
  # gradient: they receive zero cotangents and are excluded from `autograd.grad`.
  differentiable = [
      t.requires_grad
      and t.dtype not in FLOAT8_TORCH_DTYPES
      and not t.is_complex()
      for t in tensor_outs
  ]
  if not any(differentiable):
    return (
        findings.report_of(_AUTOGRAD, saved_donor_findings)
        if saved_donor_findings
        else findings.VerificationReport()
    )

  # 2. Run the registered PyTorch custom backward with normalized cotangents.
  rng = torch.Generator(device="cpu")
  rng.manual_seed(2026)
  cotangents_pt, cotangents_f32 = _build_normalized_cotangents(
      tensor_outs, differentiable, rng
  )
  try:
    with _CpuPallasInterpretDispatchMode():
      pt_grads = torch.autograd.grad(
          outputs=[t for t, d in zip(tensor_outs, differentiable) if d],
          inputs=[slot.tensor for slot in diff_slots],
          grad_outputs=[c for c, d in zip(cotangents_pt, differentiable) if d],
          allow_unused=True,
      )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return findings.report_of(
        _AUTOGRAD,
        [
            *saved_donor_findings,
            findings.VerificationFinding(
                category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_AUTOGRAD,
                op_name=op_name,
                message=(
                    f"custom PyTorch backward raised {type(exc).__name__}:"
                    f" {exc}"
                ),
            ),
        ],
    )

  shape_errors = [
      findings.VerificationFinding(
          category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
          severity=findings.Severity.ERROR,
          phase=_AUTOGRAD,
          op_name=op_name,
          message=(
              f"custom PyTorch backward for arg {slot.param_idx} returned"
              f" shape {tuple(pt_grad.shape)}, expected"
              f" {tuple(slot.tensor.shape)}"
          ),
      )
      for slot, pt_grad in zip(diff_slots, pt_grads)
      if pt_grad is not None and pt_grad.shape != slot.tensor.shape
  ]
  if shape_errors:
    return findings.report_of(_AUTOGRAD, [*saved_donor_findings, *shape_errors])

  # 3. Compare against analytical `jax.vjp` if supported, else finite differences.
  ref_jax_grads = _try_analytical_jax_vjp(
      jax_callable, diff_slots, pt_args, pt_kwargs, cotangents_pt
  )
  if ref_jax_grads is not None:
    detected = _compare_vjp_gradients(
        diff_slots,
        pt_grads,
        ref_jax_grads,
        tensor_outs,
        differentiable,
        op_name,
        rtol=rtol,
        atol=atol,
    )
  else:
    detected = _compare_finite_difference_gradients(
        jax_callable,
        diff_slots,
        pt_grads,
        pt_args,
        pt_kwargs,
        cotangents_f32,
        differentiable,
        rng,
        op_name,
        rtol=rtol,
        atol=atol,
    )
  return findings.report_of(_AUTOGRAD, [*saved_donor_findings, *detected])
