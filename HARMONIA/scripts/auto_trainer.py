import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIG ---
WATCH_FOLDER = "../data/raw/drop_zone"
RAW_DATA_FILE = "../data/raw/my_raw_dump.txt"

class AutoTrainHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or event.src_path.split("/")[-1].startswith("."):
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
        print("[AUTO] ⚙️  Converting dataset...")
        subprocess.run(["python3", "prepare_dataset.py"])

        print("[AUTO] 🧠  Training...")
        subprocess.run(["python3", "train.py"])

        print("[AUTO] 📊  Results:")
        subprocess.run(["python3", "benchmark_viewer.py"])

        print("\n[AUTO] ✅ DONE! Waiting for next file...")

if __name__ == "__main__":
    if not os.path.exists(WATCH_FOLDER):
        os.makedirs(WATCH_FOLDER)

    event_handler = AutoTrainHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()

    print(f"Auto-Trainer ON! Drop .txt files into: '{WATCH_FOLDER}'")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()