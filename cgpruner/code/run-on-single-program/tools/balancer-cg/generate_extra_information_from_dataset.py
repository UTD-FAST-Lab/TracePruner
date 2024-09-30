'''
This script reads from the "combination.csv" file, and calculates 
extra information for each edge, like the depth, 
number of incoming nodes, etc. and writes it back
to a file called "combinationWithExtraFeatures.csv"
'''


import sys
import queue
import pathlib
import json
import csv
import statistics
import random

#Input parameters (input and output files)
INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
#Some fixed constants
UNREACHABLE = -1
LARGE_GRAPH_CUTOFF = 30000 
DYNAMIC_ANALYSIS_NAME = "wiretap"

#Deafult constants
EDGE_DEPTH_DEFAULT = "1000000000"
INCOMING_EDGES_DEFAULT = "-1"
OUTGOING_EDGES_DEFAULT = "-1"
FANOUT_DEFAULT = "1000000000"
SOURCE_NODE_DEG_DEFAULT = "1000000000"
REACHABLE_FROM_MAIN_DEFAULT = "0"
NUM_PATHS_TO_THIS_DEFAULT = "0"
REPEATED_EDGES_DEFAULT = "0"
DISJOINT_PATHS_DEFAULT = "-1"
FANOUT_AVG_DEFAULT = "1000000000"
FANOUT_MIN_DEFAULT = "1000000000"

'''Represents an edge of the call graph'''
class Edge:
    def __init__(self, a, b):
        self.bytecode_offset = a
        self.dest = b
        self.depth_from_main = -1
        self.src_node_in_deg = 0
        self.dest_node_out_deg = 0
        self.dest_node_in_deg = 0
        #src_node_out_deg is trivial
        # Fanout: no. of edges from a given source node,                    
        #with the same bytecode offset as this edge.
        self.fanout = 0 
        self.avg_fanout = 0
        self.min_fanout = 0
        self.reachable_from_main = False
        self.num_paths_to_this_from_main = 0
        self.repeated_edges = 0
        self.node_disjoint_paths_from_main = 0
        self.edge_disjoint_paths_from_main = 0

'''Represents a method of the call graph'''
class Node:
    def __init__(self):
        self.edges = set()
        self.depth = -1
        self.visited = False #temporary variable.

'''Represents a call graph of a static analysis'''
class Graph:
    def __init__(self):
        self.nodes = {}
        self.edge_count = 0
        self.node_count = 0
        self.relative_node_count = 0
        self.relative_edge_count = 0
        main_methodNode = ""
        self.avg_deg = 0
        self.avg_edge_fanout = 0

'''Represents an edge in the union graph'''
class UnionEdge:
    def __init__(self, a, b,c):
        self.src = a
        self.bytecode_offset = b
        self.dest = c


def write_output(fp,csv_reader,union_edge_set,callgraphs):
    #Get the new columns and write out the whole header line
    full_header_line = (csv_reader.fieldnames +
                        get_new_column_headers(csv_reader.fieldnames))
    writer = csv.DictWriter(fp, fieldnames=full_header_line)
    writer.writeheader()

    for union_edge in union_edge_set:
        row = {} #the row to be written out  
        #First add the old columns that were read, as is
        add_old_entries_to_row(row,union_edge,callgraphs)
        #Now add the new columns
        compute_output(row,union_edge,callgraphs)
        #Finally, write out the row to the file
        writer.writerow(row)


def add_old_entries_to_row(row,union_edge,callgraphs):
    #First add out the src, bytecode and dest for the edge
    row['method'] = union_edge.src
    row['offset'] = union_edge.bytecode_offset
    row['target'] = union_edge.dest

    #Next add, for each analysis, the old bit information of whether 
    #the edge exists according to the call graph
    for analysis_name,graph in callgraphs.items():
        #Check if the edge is present in the graph
        found_edge = False
        if union_edge.src in graph.nodes: 
            for edge2 in graph.nodes[union_edge.src].edges:
                if (edge2.dest==union_edge.dest
                        and edge2.bytecode_offset==union_edge.bytecode_offset):
                    found_edge = True
                    break
            
        #Print 0 or 1 depending on if the edge is there in the graph
        if found_edge:
            row[analysis_name] = "1"
        else:
            row[analysis_name] = "0"

def get_new_column_headers(old_columns):
    '''Gets a list of the names of the new columns'''
    analysis_names = old_columns[3:]
    new_headers = []

    #There will be separate column per feature per analysis
    for analysis_name in analysis_names:
        #Don't add columns for the dynamic analysis
        if (DYNAMIC_ANALYSIS_NAME==analysis_name):
            continue
        new_headers += [analysis_name + "#depth_from_main",
                    analysis_name + "#src_node_in_deg",
                    analysis_name + "#dest_node_out_deg",
                    analysis_name + "#dest_node_in_deg",
                    analysis_name + "#src_node_out_deg",
                    analysis_name + "#repeated_edges",
                    #analysis_name + "#node_disjoint_paths_from_main",
                    #analysis_name + "#edge_disjoint_paths_from_main",
                    analysis_name + "#fanout",
                    analysis_name + "#graph_node_count",
                    analysis_name + "#graph_edge_count",
                    analysis_name + "#graph_avg_deg",
                    analysis_name + "#graph_avg_edge_fanout",
                ]

    return new_headers


def compute_output(row, union_edge, callgraphs):
    '''Write out the new computed information on edge depths, etc.
    '''
    for analysis_name,graph in callgraphs.items():
        #Don't compute anything for the dynamic analysis
        if (DYNAMIC_ANALYSIS_NAME==analysis_name):
            continue
        #Check if the union_edge is present in the graph
        edge_in_graph = None
        if union_edge.src in graph.nodes: 
            for edge2 in graph.nodes[union_edge.src].edges:
                if (edge2.dest==union_edge.dest
                        and edge2.bytecode_offset==union_edge.bytecode_offset):
                    edge_in_graph = edge2
                    break
        
        #If the union_edge exists, write attribute values calculated
        if edge_in_graph is not None:
            if edge_in_graph.depth_from_main == UNREACHABLE:
                row[analysis_name + "#depth_from_main"] = EDGE_DEPTH_DEFAULT
            else:
                row[analysis_name + "#depth_from_main"] = edge_in_graph.depth_from_main

            row[analysis_name + "#src_node_in_deg"] = edge_in_graph.src_node_in_deg
            row[analysis_name + "#dest_node_out_deg"] = len(graph.nodes[union_edge.dest].edges)
            row[analysis_name + "#dest_node_in_deg"] = edge_in_graph.dest_node_in_deg
            row[analysis_name + "#src_node_out_deg"] = len(graph.nodes[union_edge.src].edges)
            row[analysis_name + "#repeated_edges"] = edge_in_graph.repeated_edges
            #row[analysis_name + "#node_disjoint_paths_from_main"] = (
                #edge_in_graph.node_disjoint_paths_from_main)
            #row[analysis_name + "#edge_disjoint_paths_from_main"] = (
                #edge_in_graph.edge_disjoint_paths_from_main)
            row[analysis_name + "#fanout"] = edge_in_graph.fanout
 
        #Else write out the default value - 
        #(because the final table cannot have empty cells)
        else: 
            row[analysis_name + "#depth_from_main"] = EDGE_DEPTH_DEFAULT
            row[analysis_name + "#src_node_in_deg"] = INCOMING_EDGES_DEFAULT
            row[analysis_name + "#dest_node_out_deg"] = OUTGOING_EDGES_DEFAULT
            row[analysis_name + "#dest_node_in_deg"] = INCOMING_EDGES_DEFAULT
            row[analysis_name + "#src_node_out_deg"] = SOURCE_NODE_DEG_DEFAULT
            row[analysis_name + "#repeated_edges"] = REPEATED_EDGES_DEFAULT
            #row[analysis_name + "#node_disjoint_paths_from_main"] = DISJOINT_PATHS_DEFAULT
            #row[analysis_name + "#edge_disjoint_paths_from_main"] = DISJOINT_PATHS_DEFAULT
            row[analysis_name + "#fanout"] = FANOUT_DEFAULT

        #The remaining attributes are at the graph level,
        # and don't need the edge to be present
        row[analysis_name + "#graph_node_count"] = str(
            float(graph.node_count))
        row[analysis_name + "#graph_edge_count"] = str(
            float(graph.edge_count))
        row[analysis_name + "#graph_avg_deg"] = str(
            float(graph.avg_deg))
        row[analysis_name + "#graph_avg_edge_fanout"] = str(
            float(graph.avg_edge_fanout))
      

def compute_edge_depths(graph,main_method):
    '''Computes the depth of an edge 
    (defined as the depth of the source node) - using BFS
    '''
    node_depths_main = compute_bfs_node_depths(graph,[main_method])
    #Now record the edge depths and the depth of the source node of that edge
    for node_name,node_object in graph.nodes.items():
        for edge in node_object.edges:
            edge.depth_from_main = node_depths_main[node_object]


def compute_bfs_node_depths(graph,zero_depth_nodes):
    '''Just a helper function for compute_edge_depths(). The
    only reason for factoring out into a separate function is to
    avoid duplicating code
    '''
    node_depths = {}
    nodes_to_visit = queue.Queue()
    explored_set = set()
    #Initialize every zero_depth_node(root nodes) to depth 0
    #Initialize every other_node to depth -1 (unreachable)
    for node in graph.nodes:
        if node in zero_depth_nodes:
            node_depths[graph.nodes[node]] = 0
            nodes_to_visit.put(node)
            explored_set.add(node)
        else:
            node_depths[graph.nodes[node]] = UNREACHABLE

    #First compute the node depths using BFS
    while (not nodes_to_visit.empty()):
        current_node = nodes_to_visit.get()
        for edge in graph.nodes[current_node].edges:
            if edge.dest not in explored_set:
                explored_set.add(edge.dest)
                nodes_to_visit.put(edge.dest)
                node_depths[graph.nodes[edge.dest]]= (
                    node_depths[graph.nodes[current_node]] + 1)
    return node_depths


def compute_edge_reachability(graph):
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            if edge.depth_from_main!=-1:
                edge.reachable_from_main = True
            else:
                edge.reachable_from_main = False


def compute_src_node_in_deg(graph):
    '''
    For every edge 'e', just increment the 'src_node_in_deg' variable 
    of outgoing edge from the 'e.dest'
    '''
    for node in graph.nodes:
        for incomingedge in graph.nodes[node].edges:
            srcNode = incomingedge.dest
            for edge in graph.nodes[srcNode].edges:
                edge.src_node_in_deg += 1


def compute_dest_node_in_deg(graph):
    '''
    For every node, first compute the in-degree.
    Then every edge which has this as the destination node
    can be updated with it's value
    '''
    #Compute in-degree for each node
    in_degs = {}
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            if edge.dest not in in_degs:
                in_degs[edge.dest] = 0
            in_degs[edge.dest] += 1

    #Update dest_node_in_deg for each edge
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            edge.dest_node_in_deg = in_degs[edge.dest]

def compute_edge_fanouts(graph):
    '''For every edge, compute the number of edges from the same node, 
    with the same bytecode-offset
    '''
    for node in graph.nodes:
        #For 'node', first compute the number of edges at each unique 
        #bytecode-offset. This is accomplished with a Hashtable with 
        #(key=bytecode), and (value = no. of edges with same bytecode offset)
        fanout_hashtable = {} 
        for edge in graph.nodes[node].edges:
            if (edge.bytecode_offset not in fanout_hashtable):
                fanout_hashtable[edge.bytecode_offset] = 0
            fanout_hashtable[edge.bytecode_offset] += 1

        #Now update each edge with the number of edges 
        #at the same bytecode-offset as it
        for edge in graph.nodes[node].edges:
            edge.fanout = fanout_hashtable[edge.bytecode_offset]

def compute_repeated_edges(graph):
    '''For every edge, compute the number of edges from the same node,
    with the same destination node
    '''
    for node in graph.nodes:
        #For 'node', first compute the number of edges for each unique 
        #dest. This is accomplished with a Hashtable with 
        #(key=dest), and (value = no. of edges with same dest)
        dest_hashtable = {}
        for edge in graph.nodes[node].edges:
            if (edge.dest not in dest_hashtable):
                dest_hashtable[edge.dest] = 0
            dest_hashtable[edge.dest] += 1

        #Now update each edge with the number of edges 
        #with the same dest as it
        for edge in graph.nodes[node].edges:
            edge.repeated_edges = dest_hashtable[edge.dest]

'''The number of outgoing edges of an edge is 
    the number of outgoing edges from it's destination node.
def compute_dest_node_out_deg(graph):
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            edge.dest_node_out_deg = len(graph.nodes[edge.dest].edges)
'''


def compute_node_and_edge_counts(graph):
    '''Simple node and edge counts at the graph level'''
    graph.node_count = len(graph.nodes)
    for node in graph.nodes:
        for edge in graph.nodes[node].edges:
            graph.edge_count += 1


def compute_relative_node_and_edge_counts(graph,ref_graph):
    '''Computes the node and edge counts in the graph,
    relative to the reference graph'''
    graph.relative_node_count = (
        graph.node_count / ref_graph.node_count)
    graph.relative_edge_count = (
        graph.edge_count / ref_graph.node_count)

def remove_repeated_edges_from_union(union_edge_set):
    '''Removes all repeated edges in the graph.
    if e1 and e2 are edges with same src, dest and different bytecode
    offset, they are repeated edges. 1 of them will be removed.
    '''
    unique_src_dest_pairs = set()
    edges_to_remove = []
    for edge in union_edge_set:
        if (edge.src,edge.dest) in unique_src_dest_pairs: #repeated edge
            edges_to_remove.append(edge)
        else:
            unique_src_dest_pairs.add((edge.src,edge.dest))

    for edge in edges_to_remove:
        union_edge_set.remove(edge)


def compute_graph_level_info(graph):
    '''Compute some graph level information.
    Will be common to all edges.
    '''   

    #Compute the average degree of the nodes
    total_deg = 0
    total_nodes = 0
    for node_name,node_object in graph.nodes.items():
        total_nodes += 1.0
        total_deg += len(node_object.edges)
    graph.avg_deg = total_deg/total_nodes

    #Compute average edge fanout
    total_fanout = 0
    total_edges = 0
    for node_name,node_object in graph.nodes.items():
        for edge in node_object.edges:
            total_edges += 1.0
            total_fanout += edge.fanout
    graph.avg_edge_fanout = total_fanout/total_edges

def compute_node_disjoint_paths(graph,main_method):
    '''Computes the number of maximal (not the same as maximum) 
    node-disjoint paths (this is an estimate of the
    maximum node disjoint paths) to each edge,
    starting at main
    '''
    for node in graph.nodes:
        if node==main_method:
            #print(node)
            continue #skip main method
        node_disjoint_paths_from_main = 0
        nodes_used_up = set()
        #Each loop iteration looks for 1 path, 
        #and then removes the nodes on that path
        while True:
            path = find_node_disjoint_path(graph,main_method,node,nodes_used_up)
            if path==None: #No more paths exist
                break
            for n in path:
                nodes_used_up.add(n)
            node_disjoint_paths_from_main += 1
        #Add the calculated value to every outgoing edge from this node.
        for edge in graph.nodes[node].edges:
            edge.node_disjoint_paths_from_main = node_disjoint_paths_from_main

    #For edges out of main set the number of node disjoint paths as 1
    for edge in graph.nodes[main_method].edges:
        edge.node_disjoint_paths_from_main = 1

def find_node_disjoint_path(graph,start_node,dest_node,nodes_used_up):
    '''This function finds a path from the current node to 
    the destination node using BFS
    '''
    #Initialization for BFS
    nodes_to_visit = queue.Queue()
    nodes_to_visit.put(start_node)
    explored_set = {start_node}
    parent_node = {}
    for node in graph.nodes:
        parent_node[node] = None

    #Main BFS-loop
    while (not nodes_to_visit.empty()):
        current_node = nodes_to_visit.get()
        for edge in graph.nodes[current_node].edges:
            #If dest_node was found
            if current_node==dest_node:
                #Return the path by retracing the parent pointers
                path_to_dest = []
                while current_node!=None:
                    path_to_dest.append(current_node)
                    current_node = parent_node[current_node]
                return path_to_dest
            #Add the edge to the set of nodes to visit,if it has not 
            #already been explored, and is not in 'nodes_used_up'
            if (edge.dest not in explored_set 
                    and edge.dest not in nodes_used_up):
                explored_set.add(edge.dest)
                parent_node[edge.dest] = current_node
                nodes_to_visit.put(edge.dest)
    return None #No path found


def compute_edge_disjoint_paths(graph,main_method):
    '''Computes the number of maximal (not the same as maximum) 
    edge-disjoint paths (this is an estimate of the
    maximum node disjoint paths) to each edge,
    starting at main
    '''

    #For large graphs, skip this function because it takes too long
    edges_in_graph = 0
    for node in graph.nodes:
        edges_in_graph += len(graph.nodes[node].edges)
    
    if (edges_in_graph>LARGE_GRAPH_CUTOFF):
        for node in graph.nodes:
            for edge in graph.nodes[node].edges:
                edge.edge_disjoint_paths_from_main = -1
    else:
        for node in graph.nodes:
            if node==main_method:
                continue #skip main method
            edge_disjoint_paths_from_main = 0
            #Set of remaining edges. Every time a path is found,
            #the edges on the path are deleted from here
            edges_left = {} 
            for n in graph.nodes:
                edges_left[n] = set(graph.nodes[n].edges)
            #Each loop iteration looks for 1 path, 
            #and then removes the edges on that path
            while True:
                path = find_edge_disjoint_path(graph,main_method,node,edges_left)
                if path==None: #No more paths exist
                    break
                for (n,e) in path:
                    edges_left[n].remove(e)
                edge_disjoint_paths_from_main += 1
            #Add the calculated value to every outgoing edge from this node.
            for edge in graph.nodes[node].edges:
                edge.edge_disjoint_paths_from_main = edge_disjoint_paths_from_main

        #For edges out of main set the number of node disjoint paths as 1
        for edge in graph.nodes[main_method].edges:
            edge.edge_disjoint_paths_from_main = 1
        
       

def find_edge_disjoint_path(graph,start_node,dest_node,edges_left):
    '''This function finds a path from the current node to 
    the destination node using BFS
    '''
    #Initialization for BFS
    nodes_to_visit = queue.Queue()
    nodes_to_visit.put(start_node)
    explored_set = {start_node}
    parent_node_and_edge = {} #need to record parent edge as well as node
    for node in graph.nodes:
        parent_node_and_edge[node] = (None,None)

    #Main BFS-loop
    while (not nodes_to_visit.empty()):
        current_node = nodes_to_visit.get()
        #If dest_node was found
        if current_node==dest_node:
            #Return the path by retracing the parent pointers
            path_to_dest = []
            current_node,current_edge = (
                    parent_node_and_edge[dest_node])
            while current_node!=None:
                path_to_dest.append((current_node,current_edge))
                current_node,current_edge = (
                    parent_node_and_edge[current_node])
            return path_to_dest
        #Else continue BFS
        for edge in edges_left[current_node]:
            #Add the edge to the set of nodes to visit, if it has not 
            #already been explored.
            if edge.dest not in explored_set:
                explored_set.add(edge.dest)
                parent_node_and_edge[edge.dest] = (current_node,edge)
                nodes_to_visit.put(edge.dest)
    return None #No path found

def main():
    #Read the input file
    with open(INPUT_FILE, "r") as readfp:  
        #Some initialization
        callgraphs = {} #The dictionary of graphs for each of the analyses
        union_edge_set = []

        #Get the names of the analyses
        csv_reader = csv.DictReader(readfp)
        analysis_names = csv_reader.fieldnames[3:]

        #Create a graph for each analysis
        for analysis in analysis_names:
            callgraphs[analysis] = Graph()

        #Read rest of file
        for row in csv_reader:
            #Add the edge to the union call graph
            union_edge_set.append(
                UnionEdge(row['method'],row['offset'],row['target']))

            #Loop through the 0-1 bits for each analysis
            for analysis in analysis_names:
                #if true, then add the edge to the respective graph. 
                #Else do nothing.
                if row[analysis]=='1': 
                    #Create new node if it doesn't exist. Then add edge
                    if (row['method'] not in callgraphs[analysis].nodes):
                        callgraphs[analysis].nodes[row['method']] = Node()
                    if (row['target'] not in callgraphs[analysis].nodes):
                        callgraphs[analysis].nodes[row['target']] = Node()
                    callgraphs[analysis].nodes[row['method']].edges.add(
                        Edge(row['offset'],row['target']))

        main_method = "<boot>"

        #Get the node and edge counts for each graph
        for analysis_name,graph in callgraphs.items():
            compute_node_and_edge_counts(graph)

        #For each analysis, 
        for analysis_name,graph in callgraphs.items():
            compute_edge_depths(graph,main_method)
            compute_edge_reachability(graph)
            compute_src_node_in_deg(graph)
            compute_dest_node_in_deg(graph)
            compute_edge_fanouts(graph)
            compute_repeated_edges(graph)
            #compute_node_disjoint_paths(graph,main_method)
            #compute_edge_disjoint_paths(graph,main_method)
            #Record number of nodes, edges
            compute_graph_level_info(graph)

        #Write output
        with open(OUTPUT_FILE, "w") as fp:
            write_output(fp,csv_reader,union_edge_set,callgraphs)
                

if __name__ == '__main__':
    main()
