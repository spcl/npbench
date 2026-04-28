import os
import subprocess
import tempfile
from pathlib import Path
from  npbench.hardware_info.theoretical.cpu_gpu_info import get_cpu_cache_size

def build_stream():
    """Builds the STREAM benchmark with an appropriate size and returns the path to the executable"""
    
    parent_dir = Path(__file__).resolve().parent
    stream_path = parent_dir / "stream" / "stream.c"
    exe_path = parent_dir / "stream_exe"

    compiler = "gcc"

    flags = []
    
    l3_cache_size = get_cpu_cache_size()["L3"]

    stream_array_size = 0
    if l3_cache_size != "n/a":
        stream_array_size = l3_cache_size//2
    else:
        stream_array_size = 50000000
    
    defines = [f"-DSTREAM_ARRAY_SIZE={stream_array_size}", "-DNTIMES=20"]


    if compiler in ("gcc", "clang"):
        flags = ["-O3", "-march=native", "-fopenmp"]
        cmd = [compiler, stream_path, "-o", exe_path] + flags + defines

    subprocess.check_call(cmd)

    return exe_path

def get_sustained_memory_bandwidth_with_stream():
    stream_exe = build_stream()
    output = subprocess.run(
        [stream_exe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if output.returncode != 0:
        raise RuntimeError("STREAM failed:\n" + output.stderr)
    
    lines = output.stdout.split('\n')
    print(output.stdout)
    result = {}
    for line in lines:
        if line.startswith("Copy"):
            result["Copy"] = line.split()[1].strip()
        elif line.startswith("Scale"):
            result["Scale"] = line.split()[1].strip()
        elif line.startswith("Add"):
            result["Add"] = line.split()[1].strip()
        elif line.startswith("Triad"):
            result["Triad"] = line.split()[1].strip()
    return result


if __name__ == "__main__":
    print(get_sustained_memory_bandwidth_with_stream())