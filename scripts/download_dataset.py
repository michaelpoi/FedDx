
from pathlib import Path
from huggingface_hub import snapshot_download

DATASET_REPO = "aai530-group6/ddxplus"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ddxplus"


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_REPO}...")
    print(f"Destination: {DATA_DIR}")

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=["*.csv", "*.json"],
    )

    print("\nDownload complete!")
    print("\nDownloaded files:")

    for file in sorted(DATA_DIR.rglob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 ** 2)
            print(f"  {file.relative_to(DATA_DIR)} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download_dataset()
