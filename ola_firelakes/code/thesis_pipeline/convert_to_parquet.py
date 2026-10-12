import pandas as pd
import time

print("Starting CSV to parquet conversion...")
start = time.time()

csv_path = "/u/home/elwasila/firelakes/data/features/all_features_merged.csv"
parquet_path = "/u/home/elwasila/firelakes/data/features/all_features_merged.parquet"

print(f"Reading CSV: {csv_path}")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Loaded: {df.shape}")

print("Writing parquet...")
df.to_parquet(parquet_path, index=False)

elapsed = time.time() - start
print(f"Done in {elapsed:.1f} seconds")
print(f"Saved to: {parquet_path}")
