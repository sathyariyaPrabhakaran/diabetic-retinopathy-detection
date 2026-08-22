from pathlib import Path

from datasets import load_dataset


DATASET_ID = "bumbledeep/aptos"
OUTPUT_ROOT = Path("data/retina")
LABEL_DIRS = {
    0: "0_no_dr",
    1: "1_mild",
    2: "2_moderate",
    3: "3_severe",
    4: "4_proliferative",
}


def main():
    print(f"Downloading labeled dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"Rows: {len(ds)}")
    print("Columns:", ds.column_names)

    for label in LABEL_DIRS:
        (OUTPUT_ROOT / LABEL_DIRS[label]).mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        label = int(row["label_code"])
        image = row["image"]
        out = OUTPUT_ROOT / LABEL_DIRS[label] / f"aptos_{i:05d}.png"
        if not out.exists():
            image.convert("RGB").save(out, format="PNG")
        if (i + 1) % 250 == 0 or i + 1 == len(ds):
            print(f"Prepared {i + 1}/{len(ds)} images")

    print("\nDataset ready at:", OUTPUT_ROOT.resolve())
    for label, name in LABEL_DIRS.items():
        count = len(list((OUTPUT_ROOT / name).glob("*.png")))
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
