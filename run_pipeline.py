import subprocess
import sys
import os

SCRIPTS = [
    ("src/ingestion.py", "Ingestion"),
    ("src/feature_engineering.py", "Feature Engineering"),
    ("src/data_splitting.py", "Data Splitting"),
    ("src/train.py", "Training"),
    ("src/evaluate.py", "Evaluation"),
]

def run():
    root = os.path.dirname(os.path.abspath(__file__))
    for script, name in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  Running {name}...")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, os.path.join(root, script)], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ERROR in {name}:")
            print(result.stderr)
            sys.exit(1)
        print(f"  {name} completed successfully.")
    print(f"\n{'='*60}")
    print(f"  Pipeline complete! View results: mlflow ui")
    print(f"{'='*60}")

if __name__ == "__main__":
    run()
