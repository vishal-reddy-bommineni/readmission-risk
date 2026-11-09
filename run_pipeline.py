"""One command for Windows users without `make`:  python run_pipeline.py"""
import subprocess
import sys

for step in (["-m", "src.build_features"], ["-m", "src.train", "--n-boot", "1000"]):
    subprocess.run([sys.executable, *step], check=True)
print("\nNext: streamlit run src/app.py")
