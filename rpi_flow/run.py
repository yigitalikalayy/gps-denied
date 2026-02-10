import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from px4flow_rpi.main import main  # noqa: E402

raise SystemExit(main())

