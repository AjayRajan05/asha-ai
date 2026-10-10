# Copyright 2026 Ajay Rajan
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

"""Hardware detection and optional GPU simulation for AshaFit."""

from __future__ import annotations

import os
import platform
import re
import shutil
from typing import Optional

from .constants import AMD_APU_MARKERS, GPU_BANDWIDTH, _GiB
from .types import GPUInfo, HardwareProfile


def _lookup_bandwidth(gpu_name: str) -> Optional[float]:
    upper = gpu_name.upper()
    for key, bw in sorted(GPU_BANDWIDTH.items(), key=lambda x: len(x[0]), reverse=True):
        if key in upper:
            return bw
    return None


def _detect_ram_bytes() -> int:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            windll = getattr(ctypes, "windll", None)
            kernel32 = getattr(windll, "kernel32", None) if windll is not None else None
            if kernel32 is not None and kernel32.GlobalMemoryStatusEx(
                ctypes.byref(stat)
            ):
                return int(stat.ullTotalPhys)
        elif platform.system() == "Darwin":
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        else:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 16 * _GiB


def _detect_disk_free_bytes() -> int:
    try:
        return shutil.disk_usage(os.path.expanduser("~")).free
    except OSError:
        return 0


def _detect_cpu_cores() -> int:
    return os.cpu_count() or 4


def _detect_cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            import subprocess

            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _detect_avx() -> tuple[bool, bool]:
    try:
        import cpuinfo

        flags = cpuinfo.get_cpu_info().get("flags") or []
        return ("avx2" in flags, "avx512f" in flags)
    except Exception:
        return False, False


def _detect_nvidia_gpus() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            gpus.append(
                GPUInfo(
                    name=name,
                    vendor="nvidia",
                    vram_bytes=int(mem.total),
                    compute_capability=(int(cc[0]), int(cc[1])),
                    memory_bandwidth_gbps=_lookup_bandwidth(name),
                )
            )
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return gpus


def _detect_apple_gpu() -> list[GPUInfo]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    try:
        import subprocess

        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        )
        name = chip.strip()
        ram = _detect_ram_bytes()
        unified = int(ram * 0.75)
        return [
            GPUInfo(
                name=name,
                vendor="apple",
                vram_bytes=unified,
                shared_memory=True,
                memory_bandwidth_gbps=_lookup_bandwidth(name) or 400.0,
            )
        ]
    except Exception:
        return []


def _detect_windows_gpus() -> list[GPUInfo]:
    if platform.system() != "Windows":
        return []
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json",
            ],
            text=True,
        )
        import json

        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        gpus: list[GPUInfo] = []
        for item in data:
            name = str(item.get("Name") or "Unknown GPU")
            vram = int(item.get("AdapterRAM") or 0)
            if vram <= 0 or vram >= 4294967296:
                vram = 8 * _GiB
            vendor = (
                "nvidia"
                if "NVIDIA" in name.upper()
                else "amd" if "AMD" in name.upper() else "intel"
            )
            gpus.append(
                GPUInfo(
                    name=name,
                    vendor=vendor,
                    vram_bytes=vram,
                    memory_bandwidth_gbps=_lookup_bandwidth(name),
                    shared_memory=any(m in name.upper() for m in AMD_APU_MARKERS),
                )
            )
        return gpus
    except Exception:
        return []


def _simulate_gpu(gpu_name: str, vram_gb: Optional[float] = None) -> GPUInfo:
    upper = gpu_name.upper()
    vram = int((vram_gb or 24) * _GiB)
    match = re.search(r"(\d+)\s*GB", upper)
    if match:
        vram = int(float(match.group(1)) * _GiB)
    vendor = (
        "apple"
        if "APPLE" in upper
        or "M1" in upper
        or "M2" in upper
        or "M3" in upper
        or "M4" in upper
        else "nvidia"
    )
    if "AMD" in upper or "RADEON" in upper:
        vendor = "amd"
    return GPUInfo(
        name=gpu_name,
        vendor=vendor,
        vram_bytes=vram,
        memory_bandwidth_gbps=_lookup_bandwidth(gpu_name) or 400.0,
        shared_memory=vendor == "apple",
    )


def detect_hardware(
    *,
    gpu: Optional[str] = None,
    vram_gb: Optional[float] = None,
    cpu_only: bool = False,
) -> HardwareProfile:
    """Detect or simulate hardware profile."""
    os_name = platform.system().lower()
    if os_name not in ("linux", "darwin", "windows"):
        os_name = "linux"

    if gpu:
        simulated_gpus = [_simulate_gpu(gpu, vram_gb)]
        sim_vram = simulated_gpus[0].vram_bytes
        # Simulated GPU implies a matching workstation host; use real machine
        # resources only as a floor, not a cap that makes ranking empty.
        sim_ram = max(_detect_ram_bytes(), max(32 * _GiB, int(sim_vram * 1.5)))
        sim_disk = max(_detect_disk_free_bytes(), 200 * _GiB)
        return HardwareProfile(
            gpus=simulated_gpus,
            cpu_name=_detect_cpu_name(),
            cpu_cores=_detect_cpu_cores(),
            ram_bytes=sim_ram,
            disk_free_bytes=sim_disk,
            os=os_name,
            backend_hint="gguf",
            simulated=True,
        )

    detected_gpus: list[GPUInfo] = []
    if not cpu_only:
        detected_gpus.extend(_detect_nvidia_gpus())
        detected_gpus.extend(_detect_apple_gpu())
        detected_gpus.extend(_detect_windows_gpus())

    has_avx2, _has_avx512 = _detect_avx()
    backend = "gguf"
    if os_name == "linux" and detected_gpus and detected_gpus[0].vendor == "nvidia":
        backend = "gguf_or_awq"

    return HardwareProfile(
        gpus=detected_gpus,
        cpu_name=_detect_cpu_name(),
        cpu_cores=_detect_cpu_cores(),
        has_avx2=has_avx2,
        ram_bytes=_detect_ram_bytes(),
        disk_free_bytes=_detect_disk_free_bytes(),
        os=os_name,
        backend_hint=backend,
        simulated=False,
    )
