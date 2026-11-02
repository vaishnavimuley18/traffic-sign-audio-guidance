import json
from collections import Counter

JSON_PATH = "data/mapillary-traffic-signs (1).json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

features = data.get("features", [])

print("Number of features:", len(features))

categories = []

for feature in features:
    value = feature.get("properties", {}).get("value")
    if value:
        categories.append(value)

counts = Counter(categories)

print("\nTraffic Sign Categories:")
print("=" * 60)

for category, count in counts.most_common():
    print(f"{count:3d}  {category}")

print("\n" + "=" * 60)
print("Total unique categories:", len(counts))