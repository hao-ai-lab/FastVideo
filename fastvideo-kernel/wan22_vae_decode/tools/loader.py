"""Builds and loads a solution exactly as it is deployed, so that what runs here runs there.

A solution is a directory holding exactly the files that are deployed, plus `solution.json`, the deployment's build
spec, with its field names and meaning:

    {"languages": ["cuda_cpp"], "entry_point": "binding.cpp::run", "destination_passing_style": true,
     "compile_options": {"cflags": [...], "cuda_cflags": [...], "ld_flags": [...]}, "dependencies": []}

- `languages`: always ["cuda_cpp"].
- `entry_point`: "<file>::<function>", the function the deployment calls, in a C++/CUDA file.
- `destination_passing_style`: required, and true for this task: run() writes its output into `out`, its last
  argument, and returns nothing. The calling convention must never be left to inference.
- `compile_options` (optional): extra flags; see the build below. `dependencies` (optional): a list of names, which
  does not change the build. `binding` (optional): null or "torch".
Any other field is refused: the deployment would ignore it, so a misspelled field would silently fall back to its
default. Every other file in the directory is a source: non-empty UTF-8 text at a relative path, a plain file (no
links; `__pycache__` and `.pyc` files are ignored).

The module exports the entry, called positionally as the deployment calls it (argument names play no part):

    run(z, weights, rank, world, pg, out)                    every call; writes out on rank 0
    prepare(weights, rank, world, pg) -> list[Tensor]       optional, under that name: once per process, before run

(spec.py says what each argument is.) Load one solution per process.

stage(solution_dir) writes the sources into a fresh working directory and compiles every .cpp/.cc/.cxx/.c/.cu file at
its top into one extension module, in a separate process, with torch.utils.cpp_extension.load, build_directory = the
working directory, include paths [working directory, $CUTLASS_DIR/include, $CUTLASS_DIR/tools/util/include] ($CUTLASS_DIR
defaults to fastvideo-kernel/include/cutlass, the CUTLASS submodule), and these
flags on top of torch's own:
  - no compile_options: cuda_cflags [], cflags [], ld_flags ["-lcuda"];
  - compile_options given: its lists, where a missing one defaults to cflags [], cuda_cflags ["-O3",
    "--use_fast_math"], ld_flags ["-lcuda"];
  - unless a cuda_cflag names an architecture (-gencode, -arch), a -gencode flag for the GPU's own architecture with
    its arch-specific features goes first (sm_100a on compute capability 10.0), as the deployment does for its GPU.
Files in subdirectories are not compiled, but can be included. import_built(working directory) then makes it the
current directory, as the deployment does, imports the module and takes run (and prepare) from it; load() does both.
The module's name is the build's, not the solution's: PYBIND11_MODULE must be given TORCH_EXTENSION_NAME, as the
deployment builds the module under a name of its own. bench.py stages a build once and its ranks import that build.

Development builds are kept on each node, so that a job does not compile again what an earlier job on its node built:
a content-addressed cache in a directory of the node that setup/node.sh makes for the agent user (CACHE_DIR, or
$GUILD_BUILD_CACHE), keyed by everything that determines the build: the sources, solution.json, the flags, the target
architecture, this loader, and the toolchain (the machine, Python, torch with its CUDA, nvcc, the host compiler, and
the variables they read). A build found there is copied into the working directory in place of compiling it; one made
is put there with its sources and module, without what the compiler left beside them. Only the commands of a
development copy of the harness use it (tools/ in a repo: `loader.py check` and `build`, `bench.py`), unless they are
given --no-cache. The copy in the task's checks/, which CI, the hooks and the task's commands run, never reads or
writes it and builds fresh: the jobs of agents write that directory, so what it holds is theirs to forge. Past a
quarter of its filesystem, a limit that only stops a runaway, the least recently used builds are removed.

    python tools/loader.py check [solution] [--build-dir DIR]   build and load it, print what was loaded
    python tools/loader.py build [solution] [DIR]               only write and build it in DIR (default build/): with
                                                                --arch (e.g. sm_100a), on a machine without a GPU,
                                                                such as a CPU builder, where the module is not loaded
"""

import argparse
import atexit
import functools
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

SPEC_FILE = "solution.json"
# The name the build gives the module; solutions never name it themselves (see the module docstring). It is this
# kernel's own: another kernel's loader may build an extension of a generic name (solution_ext), and in one process the
# second extension of a name loads as the first one.
MODULE = "wan22_vae_decode_ext"
# The optional function a solution exports beside its entry, called once per process before any call of the entry.
PREPARE = "prepare"
BINDINGS = ("torch",)
CPP_ENTRY_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cuh")
# The fields solution.json may hold.
SPEC_FIELDS = ("languages", "entry_point", "dependencies", "destination_passing_style", "binding", "compile_options")
OPTION_FIELDS = ("cflags", "cuda_cflags", "ld_flags")
# The languages a solution may declare, and the files compiled.
LANGUAGES = ("cuda_cpp",)
CPP_SOURCES = (".cu", ".cpp", ".cc", ".cxx", ".c")
DEFAULT_OPTIONS = {"cflags": [], "cuda_cflags": ["-O3", "--use_fast_math"], "ld_flags": ["-lcuda"]}
BUILD_TIMEOUT_SEC = 1800
# The first argument of a PYBIND11_MODULE in a source.
MODULE_NAME = re.compile(r"\bPYBIND11_MODULE\s*\(\s*([A-Za-z_]\w*)")
HERE = os.path.dirname(os.path.abspath(__file__))
# The node's cache of development builds (see the module docstring), and what each build there holds beside its sources.
CACHE_DIR = "/scratch/kdev-builds"
STAMP, SIZE = ".stamp", ".size"
# The share of its filesystem past which the cache removes its least recently used builds: a limit that stops a runaway.
CACHE_SHARE = 0.25
# What a job that stopped while it stored a build left in the cache is removed after this long.
STALE_SEC = 3600
# The variables the compilers read, which change what they make.
BUILD_VARIABLES = ("CC", "CXX", "CUDA_HOME", "CUDA_PATH", "CUTLASS_DIR", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                   "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH")


class SolutionError(Exception):
    """The solution is malformed, or fails to build or load."""


def _nonempty_str(x):
    return isinstance(x, str) and len(x) >= 1


def _str_list(x):
    return isinstance(x, list) and all(isinstance(s, str) for s in x)


def spec_problems(spec):
    """Problems of a build spec, as a list of strings (empty: valid); see the module docstring.

    Stricter than the deployment only where it would coerce a value (a "true" string as a boolean) or fall back to a
    default (an unknown field, a missing destination_passing_style)."""
    if not isinstance(spec, dict):
        return [f"{SPEC_FILE} must hold an object"]
    out = []
    unknown = sorted(set(spec) - set(SPEC_FIELDS))
    if unknown:
        out.append(f"unknown fields {unknown} (known: {list(SPEC_FIELDS)})")
    langs = spec.get("languages")
    if not isinstance(langs, list) or not langs or any(lang not in LANGUAGES for lang in langs):
        out.append(f"languages: required, {list(LANGUAGES)}")
    entry = spec.get("entry_point")
    if not _nonempty_str(entry):
        out.append("entry_point: required, a non-empty string")
    elif entry.count("::") != 1:
        out.append(f'entry_point: {entry!r} is not "<file_path>::<function_name>"')
    elif os.path.splitext(entry.split("::")[0])[1] not in CPP_ENTRY_SUFFIXES:
        out.append(f"entry_point: needs a C++/CUDA entry file, not {entry.split('::')[0]!r}")
    deps = spec.get("dependencies", [])
    if not isinstance(deps, list) or not all(_nonempty_str(d) for d in deps):
        out.append("dependencies: a list of non-empty strings")
    if "destination_passing_style" not in spec:
        out.append("destination_passing_style: must be given, and true here (run writes out, its last argument)")
    elif spec["destination_passing_style"] is not True:
        out.append("destination_passing_style: true: this task's run() writes the video into out, its last argument, "
                   "and returns nothing")
    if spec.get("binding") not in (None,) + BINDINGS:
        out.append(f"binding: null or one of {list(BINDINGS)}")
    options = spec.get("compile_options")
    if options is not None:
        if not isinstance(options, dict):
            out.append("compile_options: null or an object")
        else:
            for k in OPTION_FIELDS:
                if k in options and not _str_list(options[k]):
                    out.append(f"compile_options.{k}: a list of strings")
    return out


def read_spec(solution_dir):
    """solution.json, checked against the build spec's rules (see the module docstring)."""
    path = os.path.join(solution_dir, SPEC_FILE)
    try:
        with open(path) as f:
            spec = json.load(f)
    except (OSError, ValueError) as e:
        raise SolutionError(f"cannot read {path}: {e}") from e
    problems = spec_problems(spec)
    if problems:
        raise SolutionError(f"{SPEC_FILE}: " + "; ".join(problems))
    return spec


def sources(solution_dir):
    """{relative path: text} of every file of the solution except solution.json."""
    out = {}
    for root, dirs, files in os.walk(solution_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, solution_dir).replace(os.sep, "/")
            if rel == SPEC_FILE or name.endswith(".pyc"):
                continue
            if os.path.islink(path):
                raise SolutionError(f"{rel}: sources must be plain files, not links")
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError) as e:
                raise SolutionError(f"{rel}: sources must be UTF-8 text ({e})") from e
            if not text:
                raise SolutionError(f"{rel}: sources must not be empty")
            out[rel] = text
    if not out:
        raise SolutionError(f"{solution_dir} holds no sources")
    named = sorted({m for text in out.values() for m in MODULE_NAME.findall(text) if m != "TORCH_EXTENSION_NAME"})
    if named:
        raise SolutionError(f"PYBIND11_MODULE names its module {', '.join(named)}: give it TORCH_EXTENSION_NAME, the "
                            f"name the build chooses (the deployment builds the module under a name of its own)")
    return out


def read(solution_dir):
    """(spec, {relative path: text}) of a solution, checked as stage() checks it before building: solution.json against
    the build spec's rules, the sources as the deployment takes them, the entry file among them."""
    spec = read_spec(solution_dir)
    files = sources(solution_dir)
    entry = spec["entry_point"].split("::")[0]
    if entry not in files:
        raise SolutionError(f"entry file {entry} is not among the sources")
    return spec, files


def gpu_arch():
    """The architecture solutions are built for, e.g. sm_100a: the GPU's own, with its arch-specific features. It asks
    the GPU's properties only, which makes no CUDA context."""
    import torch

    if not torch.cuda.is_available():
        raise SolutionError("there is no GPU here to take the architecture from: name it (--arch sm_100a, or the "
                            "architecture of the GPUs the build is for)")
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" + ("a" if major >= 9 else "")


def compile_flags(spec, arch):
    """(cflags, cuda_cflags, ld_flags) added to torch's own flags (see the module docstring)."""
    given = spec.get("compile_options")
    options = {"cflags": [], "cuda_cflags": [], "ld_flags": ["-lcuda"]} if given is None else {
        k: list(given.get(k, v)) for k, v in DEFAULT_OPTIONS.items()}
    if not any("-gencode" in f or "-arch" in f for f in options["cuda_cflags"]):
        options["cuda_cflags"].insert(0, f"-gencode=arch=compute_{arch[3:]},code={arch}")
    return options["cflags"], options["cuda_cflags"], options["ld_flags"]


def cache_dir():
    """The node's cache of development builds, or None where a build is made fresh: in the copy of the harness in the
    task's checks/ (CI, hooks and the task's commands run it, which must not load what a job could have put there), and
    on a node without a cache that this user may write."""
    checks = os.environ.get("CHECKS")
    if checks and os.path.realpath(checks) == os.path.realpath(HERE):
        return None
    cache = os.environ.get("GUILD_BUILD_CACHE", CACHE_DIR)
    return cache if os.path.isdir(cache) and os.access(cache, os.W_OK | os.X_OK) else None


@functools.lru_cache(maxsize=None)
def toolchain():
    """What determines a build beside the solution, its flags and the architecture: this loader, the machine, Python,
    torch with its CUDA, nvcc, the host compiler, and the variables the compilers read."""
    import torch
    from torch.utils import cpp_extension

    def version(cmd):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return (p.stdout + p.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"unavailable: {e}"

    with open(os.path.abspath(__file__), "rb") as f:
        loader = hashlib.sha256(f.read()).hexdigest()
    return {"loader": loader, "machine": platform.machine(), "python": sys.version,
            "torch": [torch.__version__, torch.version.cuda, os.path.dirname(torch.__file__)],
            "nvcc": version([os.path.join(cpp_extension.CUDA_HOME or "/usr/local/cuda", "bin", "nvcc"), "--version"]),
            "cxx": version([os.environ.get("CXX", "c++"), "--version"]),
            "variables": {k: os.environ[k] for k in BUILD_VARIABLES if k in os.environ}}


def cache_key(spec, files, flags, arch, tools):
    """The key of a build in the cache: a hash of everything that determines it (see the module docstring)."""
    what = {"spec": spec, "files": sorted(files.items()), "flags": flags, "arch": arch, "toolchain": tools}
    return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()


def disk_usage(path):
    """The bytes a directory takes on disk."""
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += getattr(st, "st_blocks", 0) * 512 or st.st_size
    return total


def fetch(cache, key, build_dir):
    """Copies the build of a key from the cache into build_dir, in its place, and marks the build used; False when the
    cache has none, or it went while it was copied."""
    entry = os.path.join(cache, key)
    if not os.path.exists(os.path.join(entry, MODULE + ".so")):
        return False
    shutil.rmtree(build_dir, ignore_errors=True)
    try:
        shutil.copytree(entry, build_dir, symlinks=True, ignore=lambda d, _: [SIZE] if d == entry else [])
        os.utime(entry)
    except OSError:
        shutil.rmtree(build_dir, ignore_errors=True)
        return False
    return True


def store(cache, key, build_dir, files):
    """Puts the build in build_dir into the cache under its key: its sources, module and stamp, without what the
    compiler left beside them. A build of the key that another job put there first stays. Then the cache is kept within
    its limit. A cache that cannot take it (a full disk, say) is left as it is: the build stands without it."""
    entry = os.path.join(cache, key)
    if os.path.exists(entry):
        return
    try:
        tmp = tempfile.mkdtemp(prefix=f".{key[:16]}-", dir=cache)
    except OSError:
        return
    try:
        for rel in [*files, MODULE + ".so", STAMP]:
            dest = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(os.path.join(build_dir, rel), dest)
        with open(os.path.join(tmp, SIZE), "w") as f:
            f.write(str(disk_usage(tmp)))
        os.rename(tmp, entry)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        return
    try:
        st = os.statvfs(cache)
        evict(cache, CACHE_SHARE * st.f_blocks * st.f_frsize)
    except OSError:
        pass


def evict(cache, limit):
    """Removes the least recently used builds of the cache until it holds at most `limit` bytes, and what jobs that
    stopped while they stored a build left there (after STALE_SEC). Anything but a directory is left alone."""
    now = time.time()
    builds = []
    for name in os.listdir(cache):
        path = os.path.join(cache, name)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode):
            continue
        if name.startswith("."):
            if now - st.st_mtime > STALE_SEC:
                shutil.rmtree(path, ignore_errors=True)
            continue
        try:
            with open(os.path.join(path, SIZE)) as f:
                size = int(f.read())
        except (OSError, ValueError):
            size = disk_usage(path)
        builds.append((st.st_mtime, path, size))
    total = sum(size for _, _, size in builds)
    for _, path, size in sorted(builds):
        if total <= limit:
            break
        shutil.rmtree(path, ignore_errors=True)
        total -= size


def stage(solution_dir, build_dir=None, arch=None, extra_cuda_cflags=(), cache=None):
    """Writes the sources into build_dir (a fresh temporary directory if None) and builds them there.

    Returns (workdir, spec, build seconds or None when an earlier build was reused). A build_dir that already holds
    a build of the same sources and flags is reused, so CI builds each solution once for its rounds. With a cache
    (cache_dir(): a development build), a build of the same key is copied from it, and one made is put there."""
    spec, files = read(solution_dir)
    arch = arch or gpu_arch()
    flags = compile_flags(spec, arch)
    # Flags a tool adds that do not change the code, such as -lineinfo for source-level profiles.
    flags[1].extend(extra_cuda_cflags)
    digest = hashlib.sha256(json.dumps([spec, flags, sorted(files.items())]).encode()).hexdigest()
    if build_dir is None:
        build_dir = tempfile.mkdtemp(prefix="solution-")
        atexit.register(shutil.rmtree, build_dir, True)
    build_dir = os.path.abspath(build_dir)
    stamp = os.path.join(build_dir, STAMP)
    try:
        with open(stamp) as f:
            if f.read() == digest and os.path.exists(os.path.join(build_dir, MODULE + ".so")):
                return build_dir, spec, None
    except OSError:
        pass
    key = cache and cache_key(spec, files, flags, arch, toolchain())
    if key and fetch(cache, key, build_dir):
        return build_dir, spec, None
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir)
    for rel, text in files.items():
        dest = os.path.join(build_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
    start = time.monotonic()
    env = dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, "-s", os.path.abspath(__file__), "_build", json.dumps(flags)]
    try:
        p = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True, timeout=BUILD_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as e:
        raise SolutionError(f"the build timed out after {BUILD_TIMEOUT_SEC} s") from e
    seconds = time.monotonic() - start
    if p.returncode != 0 or not os.path.exists(os.path.join(build_dir, MODULE + ".so")):
        log = (p.stdout + p.stderr).strip()
        raise SolutionError(f"the build failed after {seconds:.0f} s:\n{_errors(log)}")
    with open(stamp, "w") as f:
        f.write(digest)
    if key:
        store(cache, key, build_dir, files)
    return build_dir, spec, seconds


def _errors(log, limit=6000):
    """The compiler's error lines with some context, else the log's tail."""
    lines = log.splitlines()
    keep = [i for i, line in enumerate(lines) if "error" in line.lower()]
    if not keep:
        return log[-limit:]
    picked = sorted({j for i in keep for j in range(max(0, i - 2), min(len(lines), i + 3))})
    return "\n".join(lines[j] for j in picked)[-limit:]


# fastvideo-kernel's CUTLASS submodule (git submodule update --init fastvideo-kernel/include/cutlass).
_CUTLASS_SUBMODULE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "include",
                                  "cutlass")


def _build(flags):
    """The compile step, run in the working directory by stage() in its own process."""
    import torch.utils.cpp_extension as ext

    cflags, cuda_cflags, ld_flags = json.loads(flags)
    here = os.getcwd()
    srcs = sorted(os.path.join(here, p) for p in os.listdir(here)
                  if p.endswith(CPP_SOURCES) and os.path.isfile(os.path.join(here, p)))
    if not srcs:
        raise RuntimeError("No CUDA/C++ source files found in working directory")
    cutlass = os.environ.get("CUTLASS_DIR") or _CUTLASS_SUBMODULE
    try:
        ext.load(name=MODULE, sources=srcs, extra_cuda_cflags=cuda_cflags, extra_cflags=cflags, extra_ldflags=ld_flags,
                 extra_include_paths=[here, f"{cutlass}/include", f"{cutlass}/tools/util/include"],
                 build_directory=here, verbose=True)
    except ImportError as e:
        # Built, and not loadable here: a machine without a GPU has no driver to load it against, and whoever loads it
        # says whether it loads where it runs. Where a GPU is, a module that does not load (an undefined symbol, a
        # missing library) is a build that failed, and is neither stamped nor kept in the cache.
        import torch

        if torch.cuda.is_available():
            raise
        print(f"built; not loaded on this machine: {e}", flush=True)
    built = [p for p in os.listdir(here) if p.startswith(MODULE) and p.endswith(".so") and p != MODULE + ".so"]
    if built:
        os.rename(built[0], MODULE + ".so")


class Solution:
    """A loaded solution: fn, the entry (run), called as the deployment calls it, fn(*inputs, *outputs); prepare, the
    module's prepare function or None; spec (solution.json), entry, languages, language, module, workdir, and
    build_seconds (None when an earlier build was reused, or when the build was staged elsewhere)."""

    def __init__(self, spec, fn, prepare, module, workdir, build_seconds):
        self.spec = spec
        self.languages = list(spec["languages"])
        self.language = self.languages[0]
        self.entry = spec["entry_point"]
        self.fn, self.prepare, self.module = fn, prepare, module
        self.workdir, self.build_seconds = workdir, build_seconds


def import_built(workdir, spec, chdir=True, build_seconds=None):
    """Imports the module that stage() built in workdir and takes its entry and prepare from it. chdir: make the
    working directory the current one, as in deployment."""
    import torch  # noqa: F401  (the module needs libtorch loaded first)

    if chdir:
        os.chdir(workdir)
    file, func = spec["entry_point"].split("::")
    module_spec = importlib.util.spec_from_file_location(MODULE, os.path.join(workdir, MODULE + ".so"))
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as e:
        raise SolutionError(f"the built module does not load: {e}") from e
    fn = getattr(module, func, None)
    if not callable(fn):
        raise SolutionError(f"{file} defines no function {func} in its PYBIND11_MODULE")
    prepare = getattr(module, PREPARE, None)
    if prepare is not None and not callable(prepare):
        raise SolutionError(f"the module's {PREPARE} is not a function")
    return Solution(spec, fn, prepare, module, workdir, build_seconds)


def load(solution_dir, build_dir=None, chdir=True, extra_cuda_cflags=(), cache=None):
    """Builds and loads a solution as it is deployed; see the module docstring.

    build_dir: where to write and build it (a fresh temporary directory if None; reused if it holds the same build).
    chdir: make the working directory the current one, as in deployment. extra_cuda_cflags: flags a tool adds to the
    deployed ones, that must not change the code (e.g. -lineinfo). cache: the node's cache of development builds
    (cache_dir()), or None to build fresh."""
    workdir, spec, seconds = stage(solution_dir, build_dir, extra_cuda_cflags=extra_cuda_cflags, cache=cache)
    return import_built(workdir, spec, chdir=chdir, build_seconds=seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="build and load a solution")
    c.add_argument("solution", nargs="?", default="solution")
    c.add_argument("--build-dir")
    d = sub.add_parser("build", help="write and build a solution in a directory, without loading it")
    d.add_argument("solution", nargs="?", default="solution")
    d.add_argument("build_dir", nargs="?", default="build")
    d.add_argument("--arch", help="the architecture to build for, e.g. sm_100a (default: this machine's GPU's)")
    for q in (c, d):
        q.add_argument("--no-cache", action="store_true", help="build fresh, without the node's cache of builds")
    b = sub.add_parser("_build")
    b.add_argument("flags")
    a = p.parse_args()
    if a.cmd == "_build":
        _build(a.flags)
        return
    cache = None if a.no_cache else cache_dir()
    try:
        if a.cmd == "build":
            where, _, seconds = stage(os.path.abspath(a.solution), os.path.abspath(a.build_dir), a.arch, cache=cache)
            module = os.path.join(where, MODULE + ".so")
            print(f"built in {seconds:.1f} s: {module}" if seconds is not None else f"reused a build: {module}")
            return
        sol = load(os.path.abspath(a.solution), a.build_dir and os.path.abspath(a.build_dir), cache=cache)
    except SolutionError as e:
        sys.exit(f"FAILED: {e}")
    built = f", built in {sol.build_seconds:.1f} s" if sol.build_seconds is not None else ", a build reused"
    prepare = f", with {PREPARE}" if sol.prepare is not None else f", without {PREPARE}"
    print(f"loaded {sol.entry} ({', '.join(sol.languages)}){prepare}{built}")


if __name__ == "__main__":
    main()
