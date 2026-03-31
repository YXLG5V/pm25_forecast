import subprocess
import sys

START = "2023-03-02 20:00:00+00:00"
END   = "2026-03-01 18:00:00+00:00"

PIPELINE = [
    "src/01_get_pollutants_data.py",
    "src/02_get_weather_data.py",
    "src/03_preprocess.py",
    "src/04_train.py",
    "src/05_evaluate.py",
    "src/06_forecast.py"
]

def run_step(script):

    print("\n==============================")
    print(f"Running: {script}")
    print("==============================")

    result = subprocess.run([
        sys.executable,
        script,
        "--start_date", START,
        "--end_date", END
    ], check=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed at {script}")


def main():

    print("\nStarting ML pipeline\n")

    for step in PIPELINE:
        run_step(step)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()