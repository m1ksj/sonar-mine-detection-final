import os
from pathlib import Path
import sys

import uvicorn


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    uvicorn.run(
        "sonar_mine_detection.api.app:app",
        host=os.getenv("SONAR_API_HOST", "127.0.0.1"),
        port=int(os.getenv("SONAR_API_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
