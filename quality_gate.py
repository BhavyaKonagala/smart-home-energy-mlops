import json
import sys


MINIMUM_ACCURACY = 0.90


with open("metrics.json", "r") as file:
    metrics = json.load(file)


accuracy = metrics["accuracy"]

print("ML QUALITY GATE")
print("================")
print("Required Accuracy :", MINIMUM_ACCURACY)
print("Actual Accuracy   :", round(accuracy, 4))


if accuracy >= MINIMUM_ACCURACY:
    print("\nQUALITY GATE: PASSED")
    sys.exit(0)
else:
    print("\nQUALITY GATE: FAILED")
    sys.exit(1)
