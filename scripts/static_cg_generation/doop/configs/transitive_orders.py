import json
from collections import defaultdict

# Your original list of orders
orders = [
    {
        "left": "1-call-site-sensitive+heap",
        "order": "MPT",
        "right": "1-call-site-sensitive"
    },
    {
        "left": "1-object-1-type-sensitive+heap",
        "order": "MPT",
        "right": "1-object-sensitive+heap"
    },
    {
        "left": "1-object-1-type-sensitive+heap",
        "order": "MPT",
        "right": "1-type-sensitive+heap"
    },
    {
        "left": "1-object-sensitive+heap",
        "order": "MPT",
        "right": "1-object-sensitive"
    },
    {
        "left": "1-type-sensitive+heap",
        "order": "MPT",
        "right": "1-type-sensitive"
    },
    {
        "left": "1-object-sensitive+heap",
        "order": "MPT",
        "right": "1-type-sensitive+heap"
    },
    {
        "left": "1-object-sensitive",
        "order": "MPT",
        "right": "1-type-sensitive"
    },
    {
        "left": "2-call-site-sensitive+2-heap",
        "order": "MPT",
        "right": "2-call-site-sensitive+heap"
    },
    {
        "left": "2-call-site-sensitive+heap",
        "order": "MPT",
        "right": "1-call-site-sensitive+heap"
    },
    {
        "left": "2-object-sensitive+heap",
        "order": "MPT",
        "right": "1-object-1-type-sensitive+heap"
    },
    {
        "left": "2-object-sensitive+2-heap",
        "order": "MPT",
        "right": "2-object-sensitive+heap"
    },
    {
        "left": "2-object-sensitive+heap",
        "order": "MPT",
        "right": "2-type-sensitive+heap"
    },
    {
        "left": "3-object-sensitive+3-heap",
        "order": "MPT",
        "right": "2-object-sensitive+2-heap"
    },
    {
        "left": "3-object-sensitive+3-heap",
        "order": "MPT",
        "right": "3-type-sensitive+3-heap"
    },
    {
        "left": "1-call-site-sensitive",
        "order": "MPT",
        "right": "context-insensitive"
    },
    {
        "left": "1-type-sensitive",
        "order": "MPT",
        "right": "context-insensitive"
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
