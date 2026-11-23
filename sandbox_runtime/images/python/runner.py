"""Image-only entrypoint. Never import or execute this module on the host."""

import base64
import json
import os
import sys
from pathlib import Path

workspace = Path("/workspace")
request = json.loads(base64.b64decode("".join(sys.argv[1:]), validate=True))
(workspace / "artifacts").mkdir(mode=0o700)
(workspace / "inputs.json").write_text(json.dumps(request["inputs"], allow_nan=False))
(workspace / "main.py").write_text(request["code"], encoding="utf-8")
os.execv(sys.executable, [sys.executable, "-I", "-B", "/workspace/main.py"])
