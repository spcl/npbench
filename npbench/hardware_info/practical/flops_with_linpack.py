import subprocess
import shutil
import os
from pathlib import Path
import shlex
from math import sqrt
import re

def run(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def detect_mpi(mpicc="mpicc"):
    """
    Return (MPinc, MPlib) using `mpicc -show`
    """
    try:
        output = subprocess.check_output(
            [mpicc, "-show"],
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to run mpicc -show") from e

    tokens = shlex.split(output)

    mpinc_flags = []
    mplib_flags = []

    for tok in tokens:
        if tok.startswith("-I"):
            mpinc_flags.append(tok)
        elif tok.startswith("-L") or tok.startswith("-l"):
            mplib_flags.append(tok)

    mpinc = " ".join(mpinc_flags)
    mplib = " ".join(mplib_flags)

    return mpinc, mplib


def detect_openblas():
    """Return LAlib, LAinc"""
    if shutil.which("pkg-config"):
        try:
            libs = run(["pkg-config", "--libs", "openblas"])
            cflags = run(["pkg-config", "--cflags", "openblas"])
            return libs, cflags
        except subprocess.CalledProcessError:
            pass

    return "-lopenblas", ""


def detect_mkl():
    root = os.environ.get("MKLROOT")
    if not root:
        return None

    libdir = Path(root) / "lib/intel64"
    incdir = Path(root) / "include"

    return (
        f"-L{libdir} -lmkl_rt -lpthread -lm",
        f"-I{incdir}",
    )


def build_hpl(hpl_dir = Path(__file__).parent.resolve()/"linpack_hpl-2.3", arch="Linux_npbench", mpicc="mpicc"):

    print("Building HPL")
    hpl_dir = Path(hpl_dir).resolve()
    setup = hpl_dir / "setup"
    setup.mkdir(exist_ok=True)

    # --- Detect MPI ---
    mpinc, mplib = detect_mpi(mpicc)

    # --- Detect BLAS ---
    blas_lib, blas_inc = detect_openblas()

    # Optional MKL override
    mkl = detect_mkl()
    if mkl:
        blas_lib, blas_inc = mkl

    makefile = setup / f"Make.{arch}"

    makefile_text = f"""\
SHELL        = /bin/sh
CD           = cd
CP           = cp
LN_S         = ln -s
MKDIR        = mkdir -p
RM           = rm -f
TOUCH        = touch

ARCH         = {arch}

TOPdir       = {hpl_dir}
INCdir       = $(TOPdir)/include
BINdir       = $(TOPdir)/bin/$(ARCH)
LIBdir       = $(TOPdir)/lib/$(ARCH)

HPLlib       = $(LIBdir)/libhpl.a

# MPI
MPdir        =
MPinc        = {mpinc}
MPlib        = {mplib}

# BLAS
LAdir        =
LAinc        = {blas_inc}
LAlib        = {blas_lib}

# Fortran/C interface
F2CDEFS      = -DAdd_ -DF77_INTEGER=int -DStringSunStyle

# HPL includes / libs
HPL_INCLUDES = -I$(INCdir) -I$(INCdir)/$(ARCH) $(LAinc) $(MPinc)
HPL_LIBS     = $(HPLlib) $(LAlib) $(MPlib)

HPL_OPTS     =
HPL_DEFS     = $(F2CDEFS) $(HPL_OPTS) $(HPL_INCLUDES)

# Compilers
CC           = {mpicc}
CCNOOPT      = $(HPL_DEFS)
CCFLAGS      = $(HPL_DEFS) -O3 -march=native -fopenmp

LINKER       = {mpicc}
LINKFLAGS    = $(HPL_DEFS) -O3 -march=native

ARCHIVER     = ar
ARFLAGS      = r
RANLIB       = ranlib
"""
    makefile.write_text(makefile_text)

    # Build
    subprocess.run(["cp", makefile, hpl_dir], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False)
    
    arch_dir = hpl_dir / "lib" / arch

    if arch_dir.exists():
        subprocess.run(["make", "clean", f"arch={arch}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=hpl_dir,check=False)
        
    subprocess.run(["make", f"arch={arch}"], cwd=hpl_dir, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True)

    xhpl = hpl_dir / "bin" / arch / "xhpl"
    if not xhpl.exists():
        subprocess.run(["rm", hpl_dir/f"Make.{arch}"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False)
        raise RuntimeError("HPL build failed")
    
    subprocess.run(["rm", hpl_dir/f"Make.{arch}"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False)
    
    print("Successfully built HPL")

    return xhpl

def get_available_mem():
    """
    Return the best estimate of memory available to the current process.
    Priority:
      1) cgroup limit
      2) scheduler limit
      3) physical RAM
    """

    def _read_int(path):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except Exception:
            return None

    # 1) cgroup v2
    cgroup_v2 = _read_int("/sys/fs/cgroup/memory.max")
    if cgroup_v2 and cgroup_v2 > 0 and cgroup_v2 < 1 << 60:
        return cgroup_v2

    # 2) cgroup v1
    cgroup_v1 = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cgroup_v1 and cgroup_v1 > 0 and cgroup_v1 < 1 << 60:
        return cgroup_v1

    # 3) Slurm
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem:
        return int(slurm_mem) * 1024 * 1024

    slurm_mem_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_mem_cpu and slurm_cpus:
        return int(slurm_mem_cpu) * int(slurm_cpus) * 1024 * 1024

    # 4) PBS
    pbs_mem = os.environ.get("PBS_MEM")
    if pbs_mem and pbs_mem.endswith("kb"):
        return int(pbs_mem[:-2]) * 1024

    # 5) Physical RAM fallback
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024

    return None


def auto_conf_hpl(xhpl_path:Path = Path(__file__).parent.resolve()/"linpack_hpl-2.3/bin/Linux_npbench/xhpl", num_cores:int = 1, available_mem:int=get_available_mem(), NB:int=192):
    
    print("Creating HPL configuration")
    if not available_mem:
        available_mem = 17179869184

    problem_size = (int(sqrt(available_mem/8 * 0.6)) // NB) * NB

    p = 1
    q = 1
    # find closest to square grid
    core_sqrt = int(sqrt(num_cores))
    for i in reversed(range(1,core_sqrt+1)):
        if num_cores%i == 0:
            v1 = i
            v2 = num_cores//i
            p = min(v1, v2)
            q = max(v1, v2)
            break

    xhpl_dat_text= f"""HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee (this configuration was automatically created by npbench)
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
{problem_size}         Ns
1            # of NBs
192          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
{p}            Ps
{q}            Qs
16.0         threshold
1            # of panel fact
1            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
2            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
192          swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
"""

    dat_path = xhpl_path.parent.resolve()/"HPL.dat"
    with open(dat_path, 'w') as dat_file:
        dat_file.write(xhpl_dat_text)
        dat_file.close()

def run_hpl(xhpl_path = Path(__file__).parent.resolve()/"linpack_hpl-2.3/bin/Linux_npbench/xhpl", num_ranks = 1):
    print("Running HPL")

    workdir = xhpl_path.parent

    omp_threads_before = os.environ.get('OMP_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = str(1)

    output = subprocess.run(
        ["mpirun", "-np", str(num_ranks), str(xhpl_path)],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if omp_threads_before:
        os.environ['OMP_NUM_THREADS'] = omp_threads_before
    else:
        del os.environ['OMP_NUM_THREADS']

    if output.returncode != 0:
        raise RuntimeError("HPL failed:\n" + output.stderr)
    else:
        return output.stdout
        
    
def get_peak_flops(num_cores:int = 1):
    """ Builds, configures and runs the HPL benchmark, parses its output and returns the achieved GFLOPs/s
        :param num_cores: The number of cores, HPL should be configured and run with (default is 1)
    """
    
    build_hpl()
    auto_conf_hpl(num_cores=num_cores)
    hpl_output = run_hpl(num_ranks=num_cores)

    pattern = re.compile(
        r"^\s*\S+\s+"
        r"(\d+)\s+"        # N
        r"(\d+)\s+"        # NB
        r"(\d+)\s+"        # P
        r"(\d+)\s+"        # Q
        r"([\d.]+)\s+"     # Time
        r"([\deE+.-]+)",   # Gflops
        re.MULTILINE
    )

    match = pattern.search(hpl_output)

    if not match:
        raise ValueError("Could not find HPL performance line")

    N, NB, P, Q, time_s, gflops = match.groups()

    return{
        "N": int(N),
        "NB": int(NB),
        "P": int(P),
        "Q": int(Q),
        "time_seconds": float(time_s),
        "gflops": float(gflops),
    }

if __name__ == "__main__":
    import psutil
    from npbench.hardware_info.theoretical.cpu_gpu_info import get_cpu_flops
    num_cores = psutil.cpu_count(logical=False)
    hpl_results = get_peak_flops(num_cores=num_cores)

    print(
        f"""
================ HPL Results ================
Problem size: {hpl_results["N"]}
Block size: {hpl_results["NB"]}
Process grid (PxQ): {hpl_results["P"]}x{hpl_results["Q"]}
Execution time: {hpl_results["time_seconds"]}
GFLOPs: {hpl_results["gflops"]} GFLOP/s
Efficiency: {hpl_results["gflops"]/get_cpu_flops(num_cores=num_cores)[1]}
"""
    )