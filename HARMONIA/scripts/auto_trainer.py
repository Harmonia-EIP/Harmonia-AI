import time
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
WATCH_FOLDER = BASE_DIR / "data" / "raw" / "drop_zone"
RAW_DATA_FILE = BASE_DIR / "data" / "raw" / "my_raw_dump.txt"
ALLOWED_PIPELINE_SCRIPTS = {"prepare_dataset.py", "train.py", "benchmark_viewer.py"}


def run_pipeline_script(script_name):
    if script_name not in ALLOWED_PIPELINE_SCRIPTS:
        raise ValueError(f"Unexpected script name: {script_name}")

    script_path = (SCRIPTS_DIR / script_name).resolve()
    if script_path.parent != SCRIPTS_DIR.resolve() or not script_path.exists():
        raise FileNotFoundError(f"Script not found or outside scripts directory: {script_path}")

    subprocess.run([sys.executable, str(script_path)], check=True)  # nosec

class AutoTrainHandler(FileSystemEventHandler):
    def on_created(self, event):
        filename = os.path.basename(event.src_path)
        if event.is_directory or filename.startswith("."):
            return

        if not filename.lower().endswith(".txt"):
            print(f"[AUTO] Ignored non-text file: {event.src_path}")
            return

        print(f"\n[AUTO] 📂 New file detected : {event.src_path}")

        # 1. Append Data
        try:
            with open(event.src_path, "r", encoding="utf-8") as new_file:
                content = new_file.read()
            with open(RAW_DATA_FILE, "a", encoding="utf-8") as main_dump:
                main_dump.write("\n" + content)
            print(f"[AUTO] ✅ Data added to : {RAW_DATA_FILE}")
        except Exception as e:
            print(f"[AUTO] ❌ Error reading : {e}")
            return

        # 2. Run Scripts
        try:
            print("[AUTO] ⚙️  Converting dataset...")
            run_pipeline_script("prepare_dataset.py")

            print("[AUTO] 🧠  Training...")
            run_pipeline_script("train.py")

            print("[AUTO] 📊  Results:")
            run_pipeline_script("benchmark_viewer.py")
        except subprocess.CalledProcessError as e:
            print(f"[AUTO] ❌ Pipeline failed with exit code {e.returncode}")
            return

        print("\n[AUTO] ✅ DONE! Waiting for next file...")

if __name__ == "__main__":
    WATCH_FOLDER.mkdir(parents=True, exist_ok=True)
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATA_FILE.touch(exist_ok=True)

    event_handler = AutoTrainHandler()
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_FOLDER), recursive=False)
    observer.start()

    print(f"Auto-Trainer ON! Drop .txt files into: '{WATCH_FOLDER}'")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()