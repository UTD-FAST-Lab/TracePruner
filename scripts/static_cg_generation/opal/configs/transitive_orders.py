import json
from collections import defaultdict

# Your original list of orders
orders = [
          {
            "left": "11cfa",
            "order": "MPT",
            "right": "10cfa"
          },
          {
            "left": "10cfa",
            "order": "MPT",
            "right": "01cfa"
          },
          {
            "left": "01cfa",
            "order": "MPT",
            "right": "0cfa"
          },
          {
            "left": "MTA",
            "order": "MPT",
            "right": "CTA"
          },
          {
            "left": "FTA",
            "order": "MPT",
            "right": "CTA"
          }
        ]

# Build graph
graph = defaultdict(set)
for o in orders:
    graph[o["left"]].add(o["right"])

# Compute transitive closure using Floyd-Warshall-like approach
nodes = set(graph.keys()) | {r for rights in graph.values() for r in rights}
closure = defaultdict(set)

for node in nodes:
    stack = [node]
    visited = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        closure[node].add(current)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                stack.append(neighbor)

# Generate new transitive orders
existing_pairs = {(o["left"], o["right"]) for o in orders}
new_orders = []

for src in closure:
    for dst in closure[src]:
        if src != dst and (src, dst) not in existing_pairs:
            new_orders.append({
                "left": src,
                "order": "MPT",
                "right": dst
            })

# Combine
all_orders = orders + new_orders

# Optional: sort for consistency
all_orders = sorted(all_orders, key=lambda x: (x["left"], x["right"]))

# Output result
print(json.dumps({"orders": all_orders}, indent=2))
