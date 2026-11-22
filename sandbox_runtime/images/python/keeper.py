"""Image-only PID 1: bounded tmpfs lifetime even if the supervising Runtime dies."""

import math
import sys
import time

lifetime = float(sys.argv[1]) if len(sys.argv) == 2 else 35.0
if not math.isfinite(lifetime) or lifetime <= 0:
    raise SystemExit(2)
deadline = time.monotonic() + lifetime
while (remaining := deadline - time.monotonic()) > 0:
    time.sleep(min(remaining, 1.0))
# Exiting namespace PID 1 terminates every remaining container process.
