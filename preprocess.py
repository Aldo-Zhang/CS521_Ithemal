import csv
import os
import subprocess
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Step 1: Download bhive project: git clone https://github.com/ithemal/bhive.git
# Step 2: Adjust configurations and run: python3 preprocess.py

# Configuration
USE_LIMIT_SAMPLE = False
USE_PARALLELIZATION = True
SAMPLE_NUM = 10000

# File Path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BHIVE_ROOT = os.path.join(PROJECT_ROOT, "bhive")
INPUT_FILE_DICT = {
    "hsw": os.path.join(BHIVE_ROOT, "benchmark/throughput/hsw.csv"),
    "ivb": os.path.join(BHIVE_ROOT, "benchmark/throughput/ivb.csv"),
    "skl": os.path.join(BHIVE_ROOT, "benchmark/throughput/skl.csv"),
}
OUTPUT_FILE_DICT = {
    "hsw": os.path.join(PROJECT_ROOT, "bhive_training_hsw.data"),
    "ivb": os.path.join(PROJECT_ROOT, "bhive_training_ivb.data"),
    "skl": os.path.join(PROJECT_ROOT, "bhive_training_skl.data"),
}
TOKENIZER = os.path.join(PROJECT_ROOT, "data_collection/build/bin/tokenizer")


def process_row(hex_code, throughput_str):
    timing = float(throughput_str)
    try:
        xml = subprocess.check_output(
            [TOKENIZER, hex_code, "--token"], text=True
        ).strip()
        intel = subprocess.check_output(
            [TOKENIZER, hex_code, "--intel"], text=True
        ).strip()
        return (hex_code, timing, intel, xml)
    except Exception:
        return None


def main():
    print("Input Files:")
    for k, v in INPUT_FILE_DICT.items():
        print(f"  {k}: {v}")
    print("Output Files:")
    for k, v in OUTPUT_FILE_DICT.items():
        print(f"  {k}: {v}")
    print(f"Tokenizer: {TOKENIZER}")

    for arch in INPUT_FILE_DICT.keys():
        print(f"\nProcessing {arch}...")
        rows = []
        with open(INPUT_FILE_DICT[arch]) as f:
            next(f)  # skip the first line
            for hex_code, throughput_str in csv.reader(f):
                rows.append((hex_code, throughput_str))
        print(f"  Read {len(rows)} records")
        if USE_LIMIT_SAMPLE:
            rows = rows[:SAMPLE_NUM]
            print(f"  Use {len(rows)} records")

        if USE_PARALLELIZATION:
            dataset = []
            with ProcessPoolExecutor() as pool:
                futures = [pool.submit(process_row, *row) for row in rows]
                for fut in tqdm(as_completed(futures), total=len(futures)):
                    res = fut.result()
                    if res:
                        dataset.append(res)
        else:
            dataset = [process_row(*row) for row in tqdm(rows, total=len(rows))]

        torch.save(dataset, OUTPUT_FILE_DICT[arch])
        print(f"  Saved {len(dataset)} records to {OUTPUT_FILE_DICT[arch]}")


if __name__ == "__main__":
    main()
