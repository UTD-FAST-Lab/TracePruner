


import ast
import os

TRACE_DIR = '../data/encoded'

def compute_cost(s1, s2):
    """
    Compute the edit distance cost (only the last row) for converting s1 to s2.
    Returns a list of costs for aligning s1 to each prefix of s2.
    """
    prev = list(range(len(s2) + 1))
    for i in range(1, len(s1) + 1):
        curr = [i] + [0] * len(s2)
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev = curr
    return prev

def base_diff(s1, s2, i_offset=0, j_offset=0):
    """
    Compute the edit operations for small sequences using the classic DP approach.
    The offsets are added to the reported indices.
    """
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    ops = [[None]*(n+1) for _ in range(m+1)]
    
    # Base cases.
    for i in range(m+1):
        dp[i][0] = i
        ops[i][0] = 'delete' if i > 0 else None
    for j in range(n+1):
        dp[0][j] = j
        ops[0][j] = 'insert' if j > 0 else None

    # Fill the DP table.
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = 'equal'
            else:
                replace_cost = dp[i-1][j-1]
                delete_cost  = dp[i-1][j]
                insert_cost  = dp[i][j-1]
                min_cost = min(replace_cost, delete_cost, insert_cost)
                dp[i][j] = 1 + min_cost
                if min_cost == replace_cost:
                    ops[i][j] = 'replace'
                elif min_cost == delete_cost:
                    ops[i][j] = 'delete'
                else:
                    ops[i][j] = 'insert'
    
    # Traceback to get operations.
    i, j = m, n
    edits = []
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 'equal':
            edits.append(('equal', i_offset + i - 1, j_offset + j - 1, s1[i-1]))
            i -= 1
            j -= 1
        elif op == 'replace':
            edits.append(('replace', i_offset + i - 1, j_offset + j - 1, s1[i-1], s2[j-1]))
            i -= 1
            j -= 1
        elif op == 'delete':
            edits.append(('delete', i_offset + i - 1, None, s1[i-1]))
            i -= 1
        elif op == 'insert':
            edits.append(('insert', None, j_offset + j - 1, s2[j-1]))
            j -= 1
    edits.reverse()
    return edits

def hirschberg_diff(s1, s2, i_offset=0, j_offset=0):
    """
    Compute the edit operations between s1 and s2 using Hirschberg's algorithm.
    The offsets are used to track the positions relative to the original sequences.
    """
    # Base cases.
    if len(s1) == 0:
        return [('insert', None, j_offset + j, s2[j]) for j in range(len(s2))]
    if len(s2) == 0:
        return [('delete', i_offset + i, None, s1[i]) for i in range(len(s1))]
    if len(s1) == 1 or len(s2) == 1:
        return base_diff(s1, s2, i_offset, j_offset)
    
    mid = len(s1) // 2

    # Compute forward cost for s1[:mid] -> s2.
    left_cost = compute_cost(s1[:mid], s2)
    # Compute reverse cost for s1[mid:] -> s2 (reverse both sequences).
    right_cost = compute_cost(s1[mid:][::-1], s2[::-1])
    
    # Find the partition index k in s2 that minimizes the total cost.
    total_cost = [left_cost[j] + right_cost[len(s2) - j] for j in range(len(s2) + 1)]
    k = min(range(len(total_cost)), key=lambda j: total_cost[j])
    
    # Recurse on the left and right parts with updated offsets.
    left_ops = hirschberg_diff(s1[:mid], s2[:k], i_offset, j_offset)
    right_ops = hirschberg_diff(s1[mid:], s2[k:], i_offset + mid, j_offset + k)
    
    return left_ops + right_ops

def read_scg(file1, file2):
    s1 = []
    s2 = []
    file1 = os.path.join(TRACE_DIR, file1)
    file2 = os.path.join(TRACE_DIR, file2)
    with open(file1, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            # count += 1
            # if count < 100000 or count > 120000:
            #     continue
            line = line.strip()
            if not line:
                continue
            edge = ast.literal_eval(line)
            s1.append(edge)
    with open(file2, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            # count += 1
            # if count < 100000 or count > 120000:
            #     continue
            line = line.strip()
            if not line:
                continue
            edge = ast.literal_eval(line)
            s2.append(edge)
    return s1, s2

def print_diff(edits):
    for op in edits:
        if op[0] == 'equal':
            continue  # Skip printing equal parts.
        elif op[0] == 'replace':
            print(f"Replace: s1[{op[1]}]={op[3]} with s2[{op[2]}]={op[4]}")
        elif op[0] == 'delete':
            print(f"Delete: s1[{op[1]}]={op[3]}")
        elif op[0] == 'insert':
            print(f"Insert: s2[{op[2]}]={op[3]}")

if __name__ == '__main__':
    file1 = 'VC4.txt.encoded'
    file2 = 'MT1.txt.encoded'
    s1, s2 = read_scg(file1, file2)


    
    
    # Use the modified Hirschberg algorithm that carries offsets.
    edits = hirschberg_diff(s1, s2)
    print_diff(edits)

