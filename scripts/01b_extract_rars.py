"""依序解壓所有在 raw/ 但尚未在 extracted/ 的 RAR。
用 NVIDIA 附帶的 7-Zip CLI。每檔後檢查剩餘空間，< 1 GB 就停。
"""
import os
import shutil
import subprocess
import sys
import time

SEVEN_ZIP = r"C:\Program Files\NVIDIA Corporation\NVIDIA GeForce Experience\7z.exe"
RAR_DIR = r"C:\Users\alex2\Desktop\vsCode\AiJudge\data\raw"
OUT_DIR = r"C:\Users\alex2\Desktop\vsCode\AiJudge\data\extracted"
LOG = r"C:\Users\alex2\Desktop\vsCode\AiJudge\data\extract_progress.log"
MIN_FREE_GB = 1.0   # 低於此值中止避免爆盤


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def list_todo():
    months = []
    for f in sorted(os.listdir(RAR_DIR)):
        if f.endswith(".rar") and "--" in f:
            month = f.split("--")[0]
            if len(month) == 6 and month.isdigit():
                months.append((month, os.path.join(RAR_DIR, f)))
    return months


def count_files(folder):
    n = 0
    for _, _, files in os.walk(folder):
        n += len(files)
    return n


def folder_size_mb(folder):
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024 / 1024


def free_gb(path):
    _, _, free = shutil.disk_usage(path)
    return free / 1024 ** 3


def extract(rar_path, out_root):
    cmd = [SEVEN_ZIP, "x", rar_path, f"-o{out_root}", "-y", "-bd", "-mcp=950"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="cp950", errors="replace")
    return p.returncode, p.stdout, p.stderr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(f"=== extract start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    if not os.path.isfile(SEVEN_ZIP):
        log(f"7z 不存在: {SEVEN_ZIP}")
        sys.exit(1)

    todo = list_todo()
    log(f"共 {len(todo)} 個 RAR 候選（含已解的）")

    total_t0 = time.time()
    done_count = 0
    skip_count = 0
    for i, (month, rar) in enumerate(todo, 1):
        out_month = os.path.join(OUT_DIR, month)
        if os.path.exists(out_month):
            n = count_files(out_month)
            if n > 0:
                skip_count += 1
                continue
            else:
                shutil.rmtree(out_month, ignore_errors=True)

        # 空間檢查
        fg = free_gb(OUT_DIR)
        if fg < MIN_FREE_GB:
            log(f"⚠ 剩餘 {fg:.2f} GB < {MIN_FREE_GB} GB，中止以免爆盤")
            break

        size_mb = os.path.getsize(rar) / 1024 / 1024
        log(f"[{i}/{len(todo)}] {month}  開始（RAR {size_mb:.0f} MB，剩 {fg:.1f} GB）")
        t0 = time.time()
        rc, so, se = extract(rar, OUT_DIR)
        dur = time.time() - t0
        if rc != 0:
            log(f"[{i}/{len(todo)}] {month}  7z 失敗 rc={rc}")
            if se:
                log(f"  stderr: {se.strip()[:300]}")
            break
        n = count_files(out_month)
        out_mb = folder_size_mb(out_month)
        log(f"[{i}/{len(todo)}] {month}  完成（{dur:.0f}s, {n} files, {out_mb:.0f} MB）")
        done_count += 1

    total_dur = time.time() - total_t0
    log(f"=== 結束 新解 {done_count} 跳過 {skip_count} 耗時 {total_dur:.0f}s ===")


if __name__ == "__main__":
    main()
