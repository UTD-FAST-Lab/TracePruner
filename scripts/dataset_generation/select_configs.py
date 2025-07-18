import itertools
from collections import defaultdict
from typing import Dict, Tuple, List
import pandas as pd
import os


dataset_dir = "/20TB/mohammad/xcorpus-total-recall/dataset"

programs = [
    'axion',
    'batik',
    'xerces',
    'jasml'
]

tools = [
    'wala',
    'doop',
    'opal'
]


def find_best_config_subset_limited(
    unknowns: Dict[Tuple[str, str], pd.DataFrame],
    min_intersection_size: int = 300,
    max_subset_size: int = 8,
    max_configs_per_tool: int = 5,
):
    """
    unknowns: {(tool, config): DataFrame}, each DataFrame has ['method', 'offset', 'target']
    Returns the best subset of configs (across tools) satisfying tool/config limits
    """
    # Group by tool
    tool_to_configs = defaultdict(list)
    for (tool, config), df in unknowns.items():
        tool_to_configs[tool].append((config, df[['method', 'offset', 'target']].drop_duplicates()))

    # Limit to up to N configs per tool
    limited_unknowns = {}
    for tool, configs in tool_to_configs.items():
        # Optional: sort by size of df (descending) to prioritize more informative configs
        configs = sorted(configs, key=lambda x: len(x[1]), reverse=True)[:max_configs_per_tool]
        for config, df in configs:
            limited_unknowns[(tool, config)] = df

    keys = list(limited_unknowns.keys())
    best_subset = None
    best_score = float('inf')
    best_intersection = None
    best_union = None

    # Try all combinations of 2 to max_subset_size
    for r in range(2, min(max_subset_size + 1, len(keys) + 1)):
        for subset_keys in itertools.combinations(keys, r):
            # Validate tool limits
            tool_count = defaultdict(int)
            for tool, _ in subset_keys:
                tool_count[tool] += 1
            if any(count > max_configs_per_tool for count in tool_count.values()):
                continue

            dfs = [limited_unknowns[k] for k in subset_keys]

            # Compute intersection
            intersection_df = dfs[0]
            for df in dfs[1:]:
                intersection_df = pd.merge(intersection_df, df, on=['method', 'offset', 'target'], how='inner')
            intersection_df = intersection_df.drop_duplicates()

            if len(intersection_df) < min_intersection_size:
                continue

            # Compute union
            union_df = pd.concat(dfs).drop_duplicates()

            score = len(union_df) - len(intersection_df)

            if score < best_score:
                best_score = score
                best_subset = subset_keys
                best_union = union_df
                best_intersection = intersection_df

    return {
        "best_subset": best_subset,
        "union_size": len(best_union) if best_union is not None else 0,
        "intersection_size": len(best_intersection) if best_intersection is not None else 0,
        "score": best_score if best_union is not None else None,
        "intersection": best_intersection,
        "union": best_union
    }



def main():
    
    for program in programs:
        total_unknowns = {}
        
        for tool in tools:
            tool_dir = os.path.join(dataset_dir, tool, 'without_jdk', program)
            if not os.path.exists(tool_dir):
                continue

            for config in os.listdir(tool_dir):
                config_dir = os.path.join(tool_dir, config)
                if not os.path.isdir(config_dir):
                    continue

                # Read the unknowns file
                unknown_file = os.path.join(config_dir, 'unknown_edges.csv')
                if not os.path.exists(unknown_file):
                    continue
                
                unknown_df = pd.read_csv(unknown_file).drop_duplicates(subset=['method', 'offset', 'target'])

                if unknown_df.empty:
                    continue

                # total_unknowns.append(unknown_df)
                total_unknowns[(tool, config)] = unknown_df



        # Find the best subset of configs
        result = find_best_config_subset_limited(total_unknowns)
        print(f"Best subset for {program}:")
        if result['best_subset']:
            for tool, config in result['best_subset']:
                print(f"  {tool}/{config}")
            print(f"  Union size: {result['union_size']}")
            print(f"  Intersection size: {result['intersection_size']}")
            print(f"  Score: {result['score']}")
        else:
            print("  No valid subset found")
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()