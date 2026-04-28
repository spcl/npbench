import psutil
from cpuinfo import get_cpu_info
import subprocess
import json
from pathlib import Path
import os
import GPUtil
import re

def parse_lscpu():
    lscpu = subprocess.run(['lscpu'], capture_output=True, text=True)
    lscpu = lscpu.stdout.strip()
    lscpu_lines = lscpu.split('\n')

    lscpu_dict = {}

    # parse lscpu
    for line in lscpu_lines:
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            if key == "Flags":
                # Split the flags into a list of strings
                lscpu_dict[key] = value.split()
            else:
                try:
                    if '.' in value:
                        lscpu_dict[key] = float(value)
                    else:
                        lscpu_dict[key] = int(value)
                except ValueError:
                    lscpu_dict[key] = value
    return lscpu_dict

def check_sudo_privileges():
    """Check if the current user has sudo privileges by running a simple sudo command."""
    try:
        subprocess.run(['sudo', '-n', '-v'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def get_cpu_flops(num_cores):
    """Returns peak FLOPS of CPU as GFLOPs/s"""
    
    cpu_info = get_cpu_info()

    # Load specific cpu info that can only be obtained from spec sheets
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/cpu_info.json", "r") as file:
        cpu_db = json.load(file)

    cpu_vendor = cpu_info.get('vendor_id_raw', '')
    cpu_model = cpu_info.get('model', '')

    if(f"{cpu_vendor}_{cpu_model}" in cpu_db.keys()):
        #if the json with cpu info contains the current cpu, we use the info

        cpu_info = cpu_db[f"{cpu_vendor}_{cpu_model}"] | cpu_info

        # calculate peak flops as 
        # cpu_clock_frequency * core_count * num_elements_in_simd_vector * num_FMA_instructions_retired_per_cycle * 2 (1 mul 1 add)

        clock_freq = cpu_info['hz_advertised'][0]
        cpu_cores = psutil.cpu_count(logical=False)
        elements_per_vector_dp = cpu_info['SIMD width']/64
        elements_per_vector_sp = cpu_info['SIMD width']/32
        fma_tp_dp = cpu_info['DP vector FMA tp']
        fma_tp_sp = cpu_info['SP vector FMA tp']

    else:
        #else, we try to obtain as much information as possible by parsing lscpu
        lscpu_dict = parse_lscpu()

        cpu_info = lscpu_dict | cpu_info

        # Try to infer SIMD width from flags
        flags = cpu_info.get('flags', [])
        if 'avx512f' in flags:
            simd_width = 512
        elif 'avx2' in flags:
            simd_width = 256
        elif 'avx' in flags:
            simd_width = 256
        elif 'sse' in flags:
            simd_width = 128
        else:
            simd_width = 64
 
        clock_freq = cpu_info['hz_advertised'][0]
        cpu_cores = psutil.cpu_count(logical=False)
        elements_per_vector_dp = simd_width/64
        elements_per_vector_sp = simd_width/32
        fma_tp_dp = 1 # since we cannot know the fma throughput of the cpu without looking at the manual, we simply assume 1
        fma_tp_sp = 1

    cpu_cores = min(cpu_cores, num_cores)
    flops_dp = clock_freq * cpu_cores * elements_per_vector_dp * fma_tp_dp * 2/1e9
    flops_sp = clock_freq * cpu_cores * elements_per_vector_sp * fma_tp_sp * 2/1e9
        
    return flops_sp, flops_dp
        

def get_theoretical_bandwidth(num_cores = 0):
    """returns the theoretical bandwidth in GB/s"""

    #try to calculate theoretical bandwidth from dmidecode output if sudo rights available
    if check_sudo_privileges():
        dmidecode = subprocess.run(['sudo', 'dmidecode', '--type', '17'], capture_output=True, text=True)
        blocks = dmidecode.stdout.strip().split('\n\n')
        blocks = blocks[1:]
        devices = []
        for block in blocks:
            block_lines = block.split('\n')
            header = block_lines[0:2]
            block_lines = block_lines[2:]

            handle, dmi_type, structure_size = list(map(lambda x: x.strip(), header[0].split(',')))
            type_name = header[1].strip() 

            block_dict = {'Handle': handle, 'Type number': dmi_type, 'Type name': type_name, 'Structure size': structure_size}
            for block_line in block_lines:
                key, value = block_line.split(":", 1)
                block_dict[key.strip()] = value.strip()
            devices.append(block_dict)  
        used_channels = set()
        speed_mt = 0
        width = 0
        for device in devices:
            if device.get("Size", "No Module Installed") != "No Module Installed":
                used_channels.add(device.get("Bank Locator", "Unknown Channel"))
                device_speed = int(device.get("Configured Memory Speed", "0").split()[0])
                if speed_mt == 0:
                    speed_mt = device_speed
                elif speed_mt != device_speed:
                    raise NotImplementedError("The function for calculating theoretical bandwidth has not been designed to support multiple different memory speeds")

                device_width = int(device.get("Total Width", "0").split()[0])/8
                if width == 0:
                    width = device_width
                elif width != device_width:
                    raise NotImplementedError("The function for calculating theoretical bandwidth has not been designed to support multiple different memory widths")
        return speed_mt * width * len(used_channels)
    else:
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/cpu_info.json", "r") as file:
            cpu_db = json.load(file)

        lscpu_dict = parse_lscpu()
        db_key = f"{lscpu_dict.get('Vendor ID', '')}_{lscpu_dict.get('Model', '')}"
        if db_key in cpu_db:
            return lscpu_dict.get('Socket(s)', 1) * cpu_db[db_key]['Max Mem BW (GB/s)']
        else:
            # We don't know anything about the memory speed
            return 0

def get_cpu_cache_size():
    cpu_info = get_cpu_info()
    return {"L1I": cpu_info.get("l1_instruction_cache_size", "n/a"), 
            "L1D": cpu_info.get("l1_data_cache_size", "n/a"), 
            "L2": cpu_info.get("l2_cache_size", "n/a"), 
            "L3": cpu_info.get("l3_cache_size", "n/a")}


def get_gpu_flops():
    gpus = GPUtil.getGPUs()
    results = []
    for gpu in gpus:
        name = gpu.name
        cuda_cores = None
        # Try to infer CUDA cores count from GPU name if available (for NVIDIA)
        match = re.search(r'(\d{3,5})', gpu.name)
        if match:
            cuda_cores = int(match.group(1))

        # Use nvidia-smi to get more accurate info if possible
        try:
            smi_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,clocks.max.sm,clocks.current.sm', '--format=csv,noheader'],
                encoding='utf-8'
            )
            clock_speed = float(re.findall(r'(\d+)', smi_output.split(',')[1])[0]) / 1000  # GHz
        except Exception:
            clock_speed = gpu.clock / 1000  # fallback

        # Estimate FLOPs if CUDA core count known
        if cuda_cores:
            tflops = cuda_cores * clock_speed / 1e3
        else:
            tflops = None

        results.append({
            "name": name,
            "clock (GHz)": clock_speed,
            "CUDA cores": cuda_cores,
            "TFLOPs (FP32)": tflops
        })
    return results

if __name__ == "__main__":
    print("CPU flops:", get_cpu_flops(psutil.cpu_count(logical=False))[1], "GFLOPs/s")
    print(get_cpu_cache_size())
    for gpu in get_gpu_flops():
        print(f"{gpu["name"]} flops: {gpu["TFLOPs (FP32)"]} TFLOPs/s (FP32)")