import sys

def log(msg):
    with open('/u/home/elwasila/firelakes/logs/debug_nf_tsm_output.txt', 'a') as f:
        f.write(msg + '\n')
    print(msg, flush=True)

log("STEP 1: script started")

import pandas as pd
import numpy as np
import json
import os
import pickle
import gc

log("STEP 2: imports done")

DATA_DIR = "/u/home/elwasila/firelakes/data/model_ready"

log("STEP 3: reading train parquet...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_train.parquet'))
log(f"STEP 4: train_df loaded, shape={train_df.shape}")

log("STEP 5: reading val parquet...")
val_df = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_val.parquet'))
log(f"STEP 6: val_df loaded, shape={val_df.shape}")

log("STEP 7: reading test parquet...")
test_df = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_test.parquet'))
log(f"STEP 8: test_df loaded, shape={test_df.shape}")

log("STEP 9: concatenating...")
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
log(f"STEP 10: full_df concatenated, shape={full_df.shape}")

del train_df, val_df, test_df
gc.collect()
log("STEP 11: memory freed, done")
