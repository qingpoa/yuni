"""Local IDE entry point using the existing preview data directory."""

import sys
from pathlib import Path

# Direct IDE execution does not automatically include the src-layout package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from yuni_agent.config import StartupConfig, project_workspace_dir
from yuni_agent.main import run


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "output" / "first-round"
    config = StartupConfig.development().model_copy(
        update={"data_dir": root / "data", "log_dir": root / "logs", "workspace_dir": project_workspace_dir()}
    )
    run(config)


if __name__ == "__main__":
    main()
