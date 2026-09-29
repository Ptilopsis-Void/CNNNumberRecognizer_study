"""项目入口：无参数时运行完整交互流程，也支持命令行子命令。"""

from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from mnist_cnn.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
