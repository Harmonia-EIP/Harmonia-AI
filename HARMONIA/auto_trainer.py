import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIG ---
WATCH_FOLDER = "drop_zone"
RAW_DATA_FILE = "dataset/my_raw_dump.txt"

class AutoTrainHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or event.src_path.split("/")[-1].startswith("."):
            return

        print(f"\n[AUTO] 📂 New file detected : {event.src_path}")
        print("[AUTO] ⏳ Updateing dataset and re-training...")

        try:
            with open(event.src_path, "r", encoding="utf-8") as new_file:
                content = new_file.read()

            with open(RAW_DATA_FILE, "a", encoding="utf-8") as main_dump:
                main_dump.write("\n" + content)

            print(f"[AUTO] ✅ Data added to : {RAW_DATA_FILE}")
        except Exception as e:
            print(f"[AUTO] ❌ Error reading : {e}")
            return

        # 2. Exécuter la préparation (dans le dossier dataset)
        print("[AUTO] ⚙️  Converting the dataset...")
        subprocess.run(["python3", "prepare_dataset.py"], cwd="dataset")

        # 3. Exécuter l'entraînement (dans le dossier model)
        print("[AUTO] 🧠  Begining of the training...")
        subprocess.run(["python3", "train.py"], cwd="model")

        # 4. Afficher les benchmarks (à la racine)
        print("[AUTO] 📊  RESULTS :")
        subprocess.run(["python3", "benchmark_viewer.py"])

        print("\n[AUTO] ✅ DONE! Ready for the next file.")
        print(f"Waiting for file in : '{WATCH_FOLDER}'...")

if __name__ == "__main__":
    if not os.path.exists(WATCH_FOLDER):
        os.makedirs(WATCH_FOLDER)

    event_handler = AutoTrainHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()

    print(f"Auto-Trainer ON ! Drop .txt files inside the folder : '{WATCH_FOLDER}'")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()