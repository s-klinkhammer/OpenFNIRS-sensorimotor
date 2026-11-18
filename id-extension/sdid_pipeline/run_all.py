#!/usr/bin/env python3
"""
run_all.py - runs the complete pipeline in one go.

Asked once at the start (via argument OR folder dialog):
  1. BIDS folder              (--bids-root)
  2. Satori QC folders ID/SD  (--satori-id-dir, --satori-sd-dir)  [optional]
Everything else (intermediate files, latest summary/trials, quality Excel, ...)
is found automatically.

Steps:
  01 epoching -> 02/03/04 main -> 02/03/04 sensitivity
  -> 05/06/07 quality (only with Satori folders or existing rejectedChannels Excel files)
  -> 08 results document

Examples:
  python run_all.py                                   # asks for the folders via dialog
  python run_all.py --bids-root D:/BIDS --satori-id-dir D:/qc/ID --satori-sd-dir D:/qc/SD
  python run_all.py --bids-root D:/BIDS --skip-sensitivity --skip-quality
  python run_all.py --bids-root D:/BIDS --start-at 4  # continue from step 4
  python run_all.py --list                            # show the steps

If a step fails, the run stops with a hint; continue with --start-at <number>
(previous results are kept). Log: <output_root>/run_all_<timestamp>.log
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ask_folder(title):
    """Folder dialog (tkinter), falling back to terminal input. Empty/cancel -> None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        d = filedialog.askdirectory(title=title)
        root.destroy()
        return d or None
    except Exception:
        d = input(f"{title} (Enter = skip): ").strip().strip('"')
        return d or None


def build_steps(a):
    """Return the list of (number, name, group, command arguments)."""
    steps = []
    steps += [
        ("Epoching", "prep", ["01_preprocess_data.py", "--tmin", str(a.tmin), "--tmax", str(a.tmax),
                                    "--chromophore-order", a.chromophore_order]),
    ]
    for mode in (["main"] if a.skip_sensitivity else ["main", "sensitivity"]):
        grp = mode
        steps += [
            (f"Nested CV [{mode}]", grp, ["02_nested_cv.py", "--mode", mode]),
            (f"Stats report: SD vs. ID [{mode}]", grp, ["03_stats_report.py", "--mode", mode]),
            (f"Stats: group/individual vs. chance [{mode}]", grp, ["04_group_vs_chance.py", "--mode", mode]),
        ]
    if not a.skip_quality:
        if a.satori_id_dir and a.satori_sd_dir:
            steps.append(("Quality: rejected channels", "quality",
                          ["05_quality_rejected_channels.py", "--id-dir", a.satori_id_dir, "--sd-dir", a.satori_sd_dir]))
        steps += [
            ("Quality: channel labels", "quality", ["06_quality_channel_labels.py"]),
            ("Quality: group statistics", "quality", ["07_quality_group_stats.py"]),
        ]
    steps.append(("Results document", "doc", ["08_build_results_document.py"]))
    return [(i + 1, n, g, c) for i, (n, g, c) in enumerate(steps)]


def main():
    p = argparse.ArgumentParser(description="Complete SD/ID pipeline in one run",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--bids-root", help="BIDS folder (otherwise $SDID_BIDS_ROOT or dialog)")
    p.add_argument("--output-root", help="output folder (otherwise $SDID_OUTPUT_ROOT or <bids>/derivatives/sdid_pipeline)")
    p.add_argument("--satori-id-dir", help="folder with the ID rejectedChannels_SCI txt files")
    p.add_argument("--satori-sd-dir", help="folder with the SD rejectedChannels_SCI txt files")
    p.add_argument("--tmin", type=float, default=-5.0)
    p.add_argument("--tmax", type=float, default=25.0)
    p.add_argument("--chromophore-order", choices=["check", "sort"], default="check")
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--start-at", type=int, default=1, help="start at this step number")
    p.add_argument("--only", type=int, nargs="+", help="run only these step numbers")
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    p.add_argument("--no-dialog", action="store_true", help="never open dialogs or prompts")
    a = p.parse_args()

    # ---- Initial inputs ----
    bids = a.bids_root or os.environ.get("SDID_BIDS_ROOT")
    if not bids and not a.list and not a.no_dialog:
        bids = ask_folder("Select BIDS folder (containing ID/ and SD/)")
    if not bids and not a.list:
        sys.exit("No BIDS folder given (--bids-root or SDID_BIDS_ROOT).")

    if not a.skip_quality and not a.list and not a.no_dialog and not (a.satori_id_dir and a.satori_sd_dir):
        print("Quality analysis: select the Satori QC folders (cancel = use existing rejectedChannels Excel files "
              "or skip the quality part).")
        a.satori_id_dir = a.satori_id_dir or ask_folder("Satori QC folder ID (.txt)")
        a.satori_sd_dir = a.satori_sd_dir or ask_folder("Satori QC folder SD (.txt)")

    env = os.environ.copy()
    if bids:
        env["SDID_BIDS_ROOT"] = str(bids)
    if a.output_root:
        env["SDID_OUTPUT_ROOT"] = str(a.output_root)
    env["MPLBACKEND"] = "Agg"          # no plot windows that would block the run
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    # Without Satori folders, the quality part only runs if rejectedChannels Excel files already exist
    if not a.skip_quality and not (a.satori_id_dir and a.satori_sd_dir) and not a.list:
        out = Path(env.get("SDID_OUTPUT_ROOT") or (Path(bids) / "derivatives" / "sdid_pipeline"))
        qdir = out / "quality"
        have = qdir.exists() and list(qdir.glob("rejectedChannels_ID*.xlsx")) and list(qdir.glob("rejectedChannels_SD*.xlsx"))
        if not have:
            print("[INFO] No Satori folders and no existing rejectedChannels Excel files -> skipping the quality part (05-07).")
            a.skip_quality = True

    steps = build_steps(a)
    if a.list:
        for i, n, _, c in steps:
            print(f"{i:>2}  {n:<48} {' '.join(c)}")
        return

    selected = [s for s in steps if (s[0] in a.only if a.only else s[0] >= a.start_at)]

    log_dir = Path(env.get("SDID_OUTPUT_ROOT") or (Path(bids) / "derivatives" / "sdid_pipeline"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_all_{datetime.now():%Y-%m-%d_%H-%M}.log"
    print(f"BIDS:   {bids}\nOutput: {log_dir}\nLog:    {log_path}\n")

    t_all = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for i, name, _, cmd in selected:
            header = f"\n{'=' * 70}\n[{i}/{len(steps)}] {name}\n{'=' * 70}"
            print(header); log.write(header + "\n")
            t0 = time.time()
            proc = subprocess.Popen([sys.executable] + cmd, cwd=HERE, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line); log.write(line)
            rc = proc.wait()
            dt = (time.time() - t0) / 60
            msg = f"--> Step {i} ({name}) {'OK' if rc == 0 else 'FAILED (code %d)' % rc} after {dt:.1f} min"
            print(msg); log.write(msg + "\n"); log.flush()
            if rc != 0:
                print(f"\nAborted. After fixing the problem, continue with:  python run_all.py --start-at {i}"
                      + (f" --bids-root \"{bids}\"" if bids else ""))
                sys.exit(rc)

    print(f"\nFINISHED - total time {(time.time() - t_all) / 60:.1f} min. Log: {log_path}")


if __name__ == "__main__":
    main()
