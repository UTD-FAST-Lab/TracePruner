import json
from collections import defaultdict

# Your original list of orders
orders = [
    {"left": "1cfa", "order": "MPT", "right": "0cfa"},
    {"left": "2cfa", "order": "MPT", "right": "1cfa"},
    {"left": "2cfa", "order": "MPT", "right": "0cfa"},
    {"left": "VANILLA_1cfa", "order": "MPT", "right": "1cfa"},
    {"left": "VANILLA_2cfa", "order": "MPT", "right": "2cfa"},
    {"left": "VANILLA_2cfa", "order": "MPT", "right": "1cfa"},
    {"left": "VANILLA_2cfa", "order": "MPT", "right": "VANILLA_1cfa"},
    {"left": "2obj", "order": "MPT", "right": "1obj"},
    {"left": "VANILLA_1obj", "order": "MPT", "right": "1obj"},
    {"left": "VANILLA_2obj", "order": "MPT", "right": "2obj"},
    {"left": "VANILLA_2obj", "order": "MPT", "right": "VANILLA_1obj"},
    {"left": "1obj", "order": "MPT", "right": "ZEROONE_CFA"},
    {"left": "2obj", "order": "MPT", "right": "ZEROONE_CFA"},
    {"left": "1cfa", "order": "MPT", "right": "ZEROONE_CFA"},
    {"left": "2cfa", "order": "MPT", "right": "ZEROONE_CFA"},
    {"left": "ZEROONE_CFA", "order": "MPT", "right": "0cfa"},
    {"left": "ZEROONE_CONTAINER_CFA", "order": "MPT", "right": "ZEROONE_CFA"},
    {"left": "ZERO_CONTAINER_CFA", "order": "MPT", "right": "0cfa"},
    {"left": "ZEROONE_CONTAINER_CFA", "order": "MPT", "right": "ZERO_CONTAINER_CFA"},
    {"left": "VANILLA_ZEROONE_CONTAINER_CFA", "order": "MPT", "right": "ZEROONE_CONTAINER_CFA"}
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
