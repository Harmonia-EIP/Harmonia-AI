from pathlib import Path
import numpy as np


def main() -> None:
    dataset_path = Path("cleaned_dataset.npy")
    if not dataset_path.exists():
        dataset_path = Path("processed/presets.npy")
    if not dataset_path.exists():
        raise SystemExit("No dataset found. Place cleaned_dataset.npy in data/ or use data/processed/presets.npy")

    data = np.load(dataset_path, allow_pickle=True)
    print(f"Loaded: {dataset_path} | records={len(data)}")
    if len(data) > 0:
        print(data[0])


if __name__ == "__main__":
    main()
