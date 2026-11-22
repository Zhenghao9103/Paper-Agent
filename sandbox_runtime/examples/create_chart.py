import json

import matplotlib.pyplot as plt

with open("/workspace/inputs.json") as stream:
    inputs = json.load(stream)
values = inputs.get("values", [1, 2, 3, 4])
plt.plot(values, marker="o")
plt.savefig("/workspace/artifacts/chart.png")
print("Created chart.png")
