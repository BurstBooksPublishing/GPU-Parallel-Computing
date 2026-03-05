python
#!/usr/bin/env python3
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

NCU_BIN = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
if not NCU_BIN:
    sys.exit("ncu CLI not found in PATH; install Nsight Compute CLI.")

METRICS = [
    "sm__cycles_active",
    "sm__inst_executed.avg.per_cycle_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__warps_active.avg.per_cycle",
    "sm__warp_execution_efficiency",
]

def parse_args() -> List[str]:
    if len(sys.argv) < 2:
        sys.exit("Usage: ncu_profile.py <app> [args...]")
    return sys.argv[1:]

def run_ncu(app_cmd: List[str], out_csv: Path) -> None:
    cmd = [
        NCU_BIN,
        "--target-processes", "all",
        "--csv",
        "--force-overwrite",
        "--metrics", ",".join(METRICS),
        "--output", str(out_csv),
        "--",
    ] + app_cmd
    subprocess.run(cmd, check=True)

def safe_float(val: Optional[str]) -> float:
    try:
        return float((val or "").replace(",", ""))
    except ValueError:
        return 0.0

def extract_kernels(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Kernel Name")]
    if not rows:
        sys.exit("No kernel rows found in NCU CSV output.")
    return rows

def compute_metrics(k: Dict[str, str]) -> Dict[str, float]:
    cycles = safe_float(k.get("sm__cycles_active"))
    inst_pc = safe_float(k.get("sm__inst_executed.avg.per_cycle_active"))
    read = safe_float(k.get("dram__bytes_read.sum"))
    write = safe_float(k.get("dram__bytes_write.sum"))
    warps_active = safe_float(k.get("sm__warps_active.avg.per_cycle"))
    warp_eff = safe_float(k.get("sm__warp_execution_efficiency"))

    bytes_total = read + write
    flops_est = inst_pc * cycles * 32.0  # conservative vector width
    arith_int = flops_est / bytes_total if bytes_total else float("inf")
    achieved_bw_GB = (bytes_total / cycles) * 1e-6 if cycles else 0.0

    return {
        "name": k.get("Kernel Name", "unknown"),
        "cycles": cycles,
        "inst_pc": inst_pc,
        "bytes_total": bytes_total,
        "flops_est": flops_est,
        "arith_int": arith_int,
        "warp_eff": warp_eff,
        "warps_active": warps_active,
        "achieved_bw_GB": achieved_bw_GB,
    }

def print_report(m: Dict[str, float]) -> None:
    print(f"Kernel: {m['name']}")
    print(f"  cycles: {int(m['cycles'])}, inst/cycle(active): {m['inst_pc']:.3f}")
    print(f"  bytes (R+W): {m['bytes_total']:.0f}, est FLOPs: {m['flops_est']:.0f}")
    print(f"  arithmetic intensity: {m['arith_int']:.3f} FLOP/byte")
    print(f"  warp eff.: {m['warp_eff']:.1f}%, warps active: {m['warps_active']:.2f}")
    print(f"  achieved BW (bytes/cycle scaled): {m['achieved_bw_GB']:.3f} (scaled units)")
    print()

def main() -> None:
    app_cmd = parse_args()
    out_csv = Path("ncu_report.csv")
    run_ncu(app_cmd, out_csv)
    for k in extract_kernels(out_csv):
        print_report(compute_metrics(k))

if __name__ == "__main__":
    main()