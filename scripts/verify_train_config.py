"""Quick smoke check for train.py config without launching the trainer."""
import os
import sys

sys.path.insert(0, ".")

import train as t

print("--- CONFIGURED VALUES ---")
print(f"USE_CURRICULUM        = {t.USE_CURRICULUM}")
print(f"USE_SENTINEL          = {t.USE_SENTINEL}")
print(f"GEN_TEMPERATURE       = {t.GEN_TEMPERATURE}")
print(f"GEN_TOP_P             = {t.GEN_TOP_P}")
print(f"MODEL_STEPS_LIMIT     = {t.MODEL_STEPS_LIMIT}")
print(f"MAX_NEW_TOKENS        = {t.MAX_NEW_TOKENS}")
print(f"NUM_GENERATIONS       = {t.NUM_GENERATIONS}")
print(f"TRAIN_STEPS           = {t.TRAIN_STEPS}")
print(f"EVAL_MIN_DIFFICULTY   = {os.environ.get('EVAL_MIN_DIFFICULTY', '0.0')}")
print(f"CURRICULUM_OPEN_WINDOW= {os.environ.get('CURRICULUM_OPEN_WINDOW', '0')}")
print(f"MASTERY_THRESHOLD     = {os.environ.get('MASTERY_THRESHOLD', '0.70')}")
print(f"OUTPUT_DIR            = {t.OUTPUT_DIR}")
print()
print("All config values loaded cleanly.")
