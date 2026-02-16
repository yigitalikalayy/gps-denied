import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

if "--config" not in sys.argv:
    default_cfg = os.path.join(ROOT, "config_sitl.json")
    if not os.path.exists(default_cfg):
        default_cfg = os.path.join(ROOT, "config.json")
    sys.argv.extend(["--config", default_cfg])

from px4flow_rpi.main import main  # noqa: E402

raise SystemExit(main())
