"""The stack rules of a solution, checked on its sources; and the library kernels a call launches, found at run time.

The stack is CUDA C++ (with inline PTX), compiled ahead of time into one extension module (see loader.py).

Static rules (check): solution.json declares the languages in LANGUAGES, and RULES maps a rule's name to what it
looks for in the C/C++/CUDA sources and why. Kinds of match:

    include   the target of an #include
    text      the source with comments removed
    flag      solution.json: a compile option
    vendored  an #include target that must resolve to a file of the solution itself

A task's spec.py may extend the table: RULES = {name: rule, or None to drop a default rule}.

Run time (trace): one call runs under CUPTI. Every kernel is attributed to the library that launched it, from the
native call stack of its launch: kernels launched from inside cuBLAS/cuBLASLt, cuDNN or ATen (libtorch) are library
kernels, and so are Triton's (a stack the rules exclude); kernels launched from the solution's own module or by
driver calls are not, whatever their names, so kernels built from CUTLASS/CuTe templates copied into the solution
count as the solution's. Kernels whose launch cannot be traced (graph replays, device-side launches) are classified
by name. Library kernels are allowed, as placeholders: their count and their share of the call's GPU time show where
the solution still depends on them.

    python tools/audit.py [solution]    the static rules
"""

import argparse
import collections
import ctypes
import json
import os
import re
import sys

CPP_FILES = (".cu", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cuh", ".hh", ".hxx", ".inl", ".inc")
LANGUAGES = ["cuda_cpp"]

RULES = {
    "vendored-cutlass": {
        "vendored": r"(cutlass|cute)/.*",
        "why": "CUTLASS/CuTe headers are copied into the solution and maintained there (cutlass/..., cute/... next to "
               "the entry file), not taken from the installed CUTLASS"},
    "runtime-build": {
        "include": r"(.*/)?(nvrtc|nvJitLink|nvPTXCompiler)\.h",
        "text": r"\b(nvrtc[A-Z]\w*|nvJitLink[A-Z]\w*|nvPTXCompiler\w*|cuModuleLoad\w*|"
                r"cuLink(Create|AddData|AddFile|Complete)\w*|cuLibraryLoad\w*|cudaLibraryLoad\w*)\s*\(",
        "flag": r"-l(nvrtc|nvJitLink|nvptxcompiler)\w*",
        "why": "nothing is compiled or loaded at run time (NVRTC, nvJitLink, PTX or cubins loaded through the driver): "
               "device code, inline PTX included, is compiled by nvcc with the solution"},
    "python-from-cpp": {
        "text": r"\b(py(bind11)?::module_?::import|py(bind11)?::(exec|eval)(_file)?|PyImport_\w+|PyRun_\w+)\b",
        "why": "run() stays in C++ and runs no Python: Triton, torch.compile, torch.utils.cpp_extension and other "
               "kernel toolkits would come in that way"},
}


def rule_table(extra=None):
    """The rule table in force: the defaults, changed by a task's RULES (None drops a rule)."""
    table = dict(RULES)
    for name, rule in (extra or {}).items():
        if rule is None:
            table.pop(name, None)
        else:
            table[name] = rule
    return table


def _match(pattern, s):
    return re.fullmatch(pattern, s) is not None


def _strip_cpp_comments(text):
    """Removes // and /* */ comments (string literals are not parsed; good enough for these rules)."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _includes(text):
    return [(m.group(1), text.count("\n", 0, m.start()) + 1)
            for m in re.finditer(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', text, flags=re.M)]


def check(solution_dir, rules, languages=None):
    """Violations of `rules` by a solution's sources: a list of "file:line: what (rule: why)" strings.

    languages: the languages the solution may declare in solution.json (default LANGUAGES)."""
    languages = LANGUAGES if languages is None else languages
    out = []
    try:
        with open(os.path.join(solution_dir, "solution.json")) as f:
            spec = json.load(f)
    except (OSError, ValueError) as e:
        return [f"solution.json: unreadable ({e})"]
    if not isinstance(spec, dict):
        return ["solution.json: must hold an object"]
    declared = spec.get("languages")
    if not isinstance(declared, list) or not declared or any(lang not in languages for lang in declared):
        out.append(f"solution.json: languages {declared} are not allowed here (allowed: {languages})")
    options = spec.get("compile_options") if isinstance(spec.get("compile_options"), dict) else {}
    flags = [f for v in options.values() if isinstance(v, list) for f in v]
    for name, rule in rules.items():
        for flag in flags:
            if "flag" in rule and _match(rule["flag"], str(flag)):
                out.append(f"solution.json: compile option {flag!r} ({name}: {rule['why']})")
    for root, dirs, files in os.walk(solution_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for file in sorted(files):
            if not file.endswith(CPP_FILES):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            out += _check_cpp(solution_dir, os.path.relpath(path, solution_dir), text, rules)
    return out


def _check_cpp(solution_dir, rel, text, rules):
    out = []
    code = _strip_cpp_comments(text)
    includes = _includes(code)
    for name, rule in rules.items():
        for target, line in includes:
            if "include" in rule and _match(rule["include"], target):
                out.append(f"{rel}:{line}: #include <{target}> ({name}: {rule['why']})")
            if "vendored" in rule and _match(rule["vendored"], target):
                here = os.path.dirname(os.path.join(solution_dir, rel))
                if not any(os.path.isfile(os.path.join(d, target)) for d in (here, solution_dir)):
                    out.append(f"{rel}:{line}: #include <{target}> is not in the solution ({name}: {rule['why']})")
        if "text" in rule:
            for m in re.finditer(rule["text"], code):
                line = code.count("\n", 0, m.start()) + 1
                out.append(f"{rel}:{line}: {m.group(0)!r} ({name}: {rule['why']})")
                break
    return out


# Run-time audit.

# The library a kernel was launched from, by the path of the innermost library on the launch's native stack.
LAUNCHERS = [
    (r".*/libcublas(Lt)?[^/]*", "cuBLAS"),
    (r".*/libcudnn[^/]*", "cuDNN"),
    (r".*/lib(torch|c10)[^/]*", "ATen"),
    (r".*triton.*", "Triton"),
    (r".*/lib(cufft|curand|cusparse|cusolver|cutensor|nccl|nvshmem|npp)[^/]*", "CUDA library"),
]
# Frames skipped on the way out: CUPTI and this module's callback, the Python interpreter, the CUDA runtime/driver.
SKIP = r"(.*/)?(libffi[^/]*|_ctypes[^/]*|python[^/]*|libpython[^/]*|callback_wrappers[^/]*|cupti[^/]*|libcupti[^/]*|" \
       r"libcuda\.so[^/]*|libcudart[^/]*|libc\.so[^/]*|ld-linux[^/]*|\?)"
# Fallback by kernel name, for launches that could not be traced.
NAMES = [
    (r"(void )?at::.*|.*at::native::.*|_ZN2at.*|.*at_cuda_detail.*|_ZN3c10.*", "ATen"),
    (r".*(nvjet|cublas|gemv2?[NT]?_|gemmSN|splitKreduce|xmma|sgemm|hgemm|dgemm).*|cutlass3x_.*|cutlass_\d.*", "cuBLAS"),
    (r".*cudnn.*", "cuDNN"),
    (r"triton_.*", "Triton"),
]

_libc = None
_demangler = None


class _DlInfo(ctypes.Structure):
    _fields_ = [("fname", ctypes.c_char_p), ("fbase", ctypes.c_void_p), ("sname", ctypes.c_char_p),
                ("saddr", ctypes.c_void_p)]


def _launcher():
    """Path of the innermost library on the current native stack that is not CUPTI, Python or the CUDA runtime."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6")
        _libc.backtrace.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        _libc.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    buf = (ctypes.c_void_p * 96)()
    n = _libc.backtrace(buf, 96)
    for i in range(n):
        info = _DlInfo()
        path = info.fname.decode() if buf[i] and _libc.dladdr(buf[i], ctypes.byref(info)) and info.fname else "?"
        if not re.fullmatch(SKIP, path):
            return path
    return ""


def demangle(name):
    global _demangler
    if _demangler is None:
        try:
            lib = ctypes.CDLL("libstdc++.so.6")
            lib.__cxa_demangle.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_int)]
            lib.__cxa_demangle.restype = ctypes.c_char_p
            _demangler = lib
        except OSError:
            _demangler = False
    if not _demangler or not name.startswith("_Z"):
        return name
    status = ctypes.c_int()
    out = _demangler.__cxa_demangle(name.encode(), None, None, ctypes.byref(status))
    return out.decode() if status.value == 0 and out else name


def short_name(name, limit=110):
    """A kernel's demangled name without its return type, parameter list and the at::native/anonymous namespaces."""
    text = demangle(name).replace("(anonymous namespace)::", "")
    depth, start, end = 0, 0, len(text)
    for i, c in enumerate(text):
        if c in "<(":
            if c == "(" and depth == 0 and i > 0:
                end = i
                break
            depth += 1
        elif c in ">)":
            depth -= 1
        elif c == " " and depth == 0:
            start = i + 1
    text = text[start:end] or text
    for noise in ("at::native::", "at::", "std::"):
        text = text.replace(noise, "")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def classify(name, launcher, traced):
    """The library a kernel belongs to (None: the solution's own)."""
    if traced:
        for pattern, label in LAUNCHERS:
            if re.fullmatch(pattern, launcher):
                return label
        return None
    for pattern, label in NAMES:
        if re.fullmatch(pattern, name) or re.fullmatch(pattern, demangle(name)):
            return label
    return None


Op = collections.namedtuple("Op", "kind name bytes launcher library ns")


def trace(fn, *args):
    """Calls fn(*args) once under CUPTI; returns its GPU operations in start order, as Op(kind, name, bytes,
    launcher, library, ns): kind is kernel, memcpy or memset; library is None for the solution's own kernels; ns is
    the operation's GPU time."""
    import torch
    from cupti import cupti

    launches, records, errors = {}, [], []

    def on_api(userdata, domain, cbid, data):
        try:
            if data.callback_site == cupti.ApiCallbackSite.API_ENTER and "Launch" in data.function_name:
                graph = "Graph" in data.function_name
                launches[data.correlation_id] = "" if graph else _launcher()
        except Exception as e:  # never let an exception cross into the driver
            errors.append(repr(e))

    def requested():
        return 8 * 1024 * 1024, 0

    def completed(activities):
        for a in activities:
            if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                records.append((a.start, a.end, "kernel", a.name, 0, a.correlation_id,
                                bool(a.graph_id or a.is_device_launched)))
            elif a.kind == cupti.ActivityKind.MEMCPY:
                records.append((a.start, a.end, "memcpy", "", a.bytes, a.correlation_id, False))
            elif a.kind == cupti.ActivityKind.MEMSET:
                records.append((a.start, a.end, "memset", "", a.bytes, a.correlation_id, False))

    kinds = (cupti.ActivityKind.CONCURRENT_KERNEL, cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET)
    torch.cuda.synchronize()
    sub = cupti.subscribe(on_api, 0)
    try:
        cupti.enable_domain(1, sub, cupti.CallbackDomain.DRIVER_API)
        for k in kinds:
            cupti.activity_enable(k)
        cupti.activity_register_callbacks(requested, completed)
        fn(*args)
        torch.cuda.synchronize()
        cupti.activity_flush_all(1)
    finally:
        for k in kinds:
            cupti.activity_disable(k)
        cupti.unsubscribe(sub)
        cupti.finalize()
    if errors:
        raise RuntimeError(f"the launch trace failed: {errors[0]}")
    ops = []
    for start, end, kind, name, nbytes, corr, untraceable in sorted(records):
        launcher = launches.get(corr)
        traced = kind == "kernel" and not untraceable and bool(launcher)
        library = classify(name, launcher or "", traced) if kind == "kernel" else None
        ops.append(Op(kind, name, nbytes, os.path.basename(launcher or ""), library, end - start))
    return ops


def signature(ops):
    """What a call launched, for comparing calls: a multiset of (kind, name, bytes)."""
    return collections.Counter((op.kind, op.name, op.bytes) for op in ops)


def library_counts(ops):
    """{library: launches} of a call's library kernels."""
    return dict(collections.Counter(op.library for op in ops if op.library))


def library_share(ops):
    """The share of a call's GPU time (the summed time of its operations) spent in library kernels."""
    total = sum(op.ns for op in ops)
    return sum(op.ns for op in ops if op.library) / total if total else 0.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import spec  # noqa: E402

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("solution", nargs="?", default="solution")
    a = p.parse_args()
    table = rule_table(getattr(spec, "RULES", None))
    found = check(a.solution, table)
    for v in found:
        print(v)
    print(f"stack rules: {'ok' if not found else f'{len(found)} violations'} ({len(table)} rules)")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
