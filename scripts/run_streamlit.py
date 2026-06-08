from pathlib import Path
import subprocess
import sys


def main():
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "sonar_mine_detection" / "ui" / "streamlit_app.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
        ],
        cwd=repo_root,
        check=True,
    )


if __name__ == "__main__":
    main()
