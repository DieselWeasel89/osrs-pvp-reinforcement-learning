import os
import subprocess
import datetime
from pathlib import Path

# --- CONFIGURATION ---
ib_path = r"C:\Program Files (x86)\IncrediBuild\ib_console.exe"
project_dir = Path("projects")
log_dir = Path("build_logs")
config = "Release"

# Create log folder
log_dir.mkdir(exist_ok=True)

# Find .sln, .bat, .py
project_files = list(project_dir.rglob("*.sln")) + list(project_dir.rglob("*.bat")) + list(project_dir.rglob("*.py"))

if not project_files:
    print(f"❌ No buildable projects in {project_dir}")
    exit(1)

for proj in project_files:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"build-{proj.stem}-{timestamp}.log"
    ext = proj.suffix.lower()

    print(f"🚀 Building: {proj}")
    if ext == ".sln":
        cmd = [ib_path, "BuildConsole", str(proj), f"/cfg={config}", "/rebuild", "/openmonitor"]
    elif ext == ".bat":
        cmd = [ib_path, "BuildConsole", str(proj), "/openmonitor"]
    elif ext == ".py":
        cmd = [ib_path, "BuildConsole", f"python {proj}", "/openmonitor"]
    else:
        print(f"⚠️ Skipping unsupported file: {proj}")
        continue

    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, shell=False)

    print(f"✅ Log saved: {log_file}")