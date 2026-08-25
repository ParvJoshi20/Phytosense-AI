import subprocess
from pathlib import Path
from PIL import Image
import hashlib

CLASSES = {
    "Tomato leaf": "healthy",
    "Tomato leaf bacterial spot": "bacterial_spot",
    "Tomato Early blight leaf": "early_blight",
    "Tomato leaf late blight": "late_blight",
    "Tomato mold leaf": "leaf_mold",
    "Tomato Septoria leaf spot": "septoria_leaf_spot",
    "Tomato leaf mosaic virus": "mosaic_virus",
    "Tomato leaf yellow virus": "yellow_leaf_curl_virus",
}

SPLITS = ["train", "test"]

REPO = Path("data/raw/plantdoc")
OUTPUT = Path("data/processed/plantdoc_tomato")


def git_paths():
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def get_blob(path):
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    return result.stdout


def safe_name(name):
    invalid = '<>:"/\\|?*'
    for c in invalid:
        name = name.replace(c, "_")

    name = name.rstrip(" .")

    if not name:
        name = "image"

    return name


def main():
    paths = git_paths()

    total = 0
    valid = 0
    invalid = 0

    for split in SPLITS:
        for source_class, target_class in CLASSES.items():

            prefix = f"{split}/{source_class}/"

            files = [
                p for p in paths
                if p.startswith(prefix)
            ]

            output_dir = OUTPUT / split / target_class
            output_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n{split}/{target_class}: "
                f"{len(files)} files found"
            )

            for path in files:
                total += 1

                try:
                    data = get_blob(path)

                    original_name = Path(path).name
                    filename = safe_name(original_name)

                    # Prevent duplicate filenames
                    destination = output_dir / filename

                    if destination.exists():
                        h = hashlib.sha1(
                            path.encode()
                        ).hexdigest()[:8]

                        destination = (
                            output_dir /
                            f"{destination.stem}_{h}{destination.suffix}"
                        )

                    destination.write_bytes(data)

                    # Verify that the extracted file is a valid image
                    with Image.open(destination) as img:
                        img.verify()

                    valid += 1

                except Exception:
                    invalid += 1

                    if 'destination' in locals() and destination.exists():
                        destination.unlink()

    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Total files found : {total}")
    print(f"Valid images      : {valid}")
    print(f"Invalid/skipped   : {invalid}")
    print(f"Output            : {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()