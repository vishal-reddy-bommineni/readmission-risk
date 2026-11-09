"""One command for the whole pipeline (works on Windows without `make`):  python run_pipeline.py"""
import subprocess
import sys

steps = [
    ["-m", "src.build_features"],
    ["-m", "scripts.run_eda"],
    ["-m", "src.train", "--n-boot", "1000"],
]
for step in steps:
    subprocess.run([sys.executable, *step], check=True)
print("\nOptional: python -m scripts.feature_experiment")
print("Next:     streamlit run src/app.py")
