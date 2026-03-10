import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

if "--config" not in sys.argv:
    default_cfg = os.path.join(REPO_ROOT, "config.json")
    sys.argv.extend(["--config", default_cfg])

from algorithms.optical_flow.main import main  # noqa: E402

raise SystemExit(main())
