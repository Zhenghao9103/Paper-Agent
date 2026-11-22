import json

import numpy as np

with open("/workspace/inputs.json") as stream:
    inputs = json.load(stream)
print(float(np.mean(inputs.get("values", [1, 2, 3, 4]))))
