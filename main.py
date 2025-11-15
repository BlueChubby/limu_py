from datetime import datetime
from pathlib import Path

if __name__ == '__main__':

    print(Path.cwd())

    path: Path = Path(Path.cwd() / "logs")
    path.mkdir(parents=True, exist_ok=True)

    print(path)


