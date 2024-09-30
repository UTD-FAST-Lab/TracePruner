#!/usr/bin/env python3
'''
This script removes all standard library edges,
and adds those application edges which are in the transitive of the removed edges
'''
import sys
import os
import csv
from pathlib import Path
import queue
import json
from collections import defaultdict, namedtuple

ROOT_METHOD = "<boot>"

#Edge and Node classes to model the graph
Edge = namedtuple("Edge", "bytecodeOffset dest")
Node = namedtuple("Node",
            ["edges","reachable_app_nodes","reachableNodes","isStdLibNode"])

def main():
    """ The main method of the removestdlibedges.py script. """
    analysisfile = Path(sys.argv[1])
    methodsfile = Path(sys.argv[2])
    outputfile = Path(sys.argv[3])
    
    #Loop through all the file names
    nodes_with_closure = remove_stdlib_edges(analysisfile, methodsfile, True)
    nodes_without_closure = remove_stdlib_edges(analysisfile, methodsfile, False)

    #Compute without-closure edges
    without_closure_edges = set()
    for name, obj in nodes_without_closure.items():
            for edge in obj.edges:
                edge_id = (name, edge.bytecodeOffset, edge.dest)
                without_closure_edges.add(edge_id)

    #Write output
    printed = set()
    with open(outputfile, "w") as filep:
        writer = csv.writer(filep)
        writer.writerow(["method", "offset", "target","wala","wala-direct"])
        for name, obj in nodes_with_closure.items():
            for edge in obj.edges:
                edge_id = (name, edge.bytecodeOffset, edge.dest)
                if edge_id in printed:
                    continue
                printed.add(edge_id)

                if edge_id in without_closure_edges:
                    writer.writerow((name, edge.bytecodeOffset, edge.dest,"1","1"))
                else:
                    writer.writerow((name, edge.bytecodeOffset, edge.dest,"1","0"))
                

def empty_node():
    """ Create an empty node """
    return Node([], set(), set(), False)

def remove_stdlib_nodes(nodes, stdlibnodes):
    """ remove standard library nodes """
    for node in stdlibnodes:
        del nodes[node]
    return nodes

def get_std_lib_nodes_directly_called(static_analysis_nodes, std_lib_nodes):
    """ Computes the set of stdLib nodes which get directly called
    from an application node
    (does not include stdlib nodes with no outgoing edges) """
    directly_called_nodes = set()
    for node_object in static_analysis_nodes.values():
        for edge in node_object.edges:
            if edge.dest in std_lib_nodes:
                directly_called_nodes.add(edge.dest)
    return directly_called_nodes


def compute_reachable_application_nodes(std_lib_nodes,
        application_method_list, static_analysis_nodes):
    """compute the set of reachable application nodes for all the std_lib_nodes
    , by computing its transitive closure transitive closure for each nodes is
    stored in the 'reachable_app_nodes' field """
    std_lib_nodes_called_directly_from_application = \
        get_std_lib_nodes_directly_called(static_analysis_nodes, std_lib_nodes)

    #Compute reachability information for all stdlib nodes
    #called directly called from the application
    for node_name in std_lib_nodes_called_directly_from_application:
        #BFS
        nodes_to_be_explored = queue.Queue()
        visited_list = set()
        visited_list.add(node_name)
        nodes_to_be_explored.put(node_name)
        #BFS on 'node_name' to compute reachable nodes
        while not nodes_to_be_explored.empty():
            node = nodes_to_be_explored.get()
            if node in std_lib_nodes:
                for edge in std_lib_nodes[node].edges:
                    if edge.dest not in visited_list:
                        visited_list.add(edge.dest)
                        nodes_to_be_explored.put(edge.dest)
        
        #Now compute which are the application nodes from this explored list
        for node in visited_list:
            if node in application_method_list:
                std_lib_nodes[node_name].reachable_app_nodes.add(node)
    return std_lib_nodes


def get_std_lib_nodes(static_analysis_nodes, application_method_list):
    """ Gets a list of standard library nodes """
    std_lib_nodes = {}
    #if a node does not contain a class name from any of the classes
    # given by javaq, it is considered a standard library node
    for node_name, node_object in static_analysis_nodes.items():
        if node_name not in application_method_list:
            std_lib_nodes[node_name] = node_object
    return std_lib_nodes

def replace_std_lib_edges_with_app_edges(
        static_analysis_nodes, std_lib_nodes, application_method_list):
    """ Replaces edges ending in stdLib nodes with their reachable application
    nodes Assumes that wala nodes does not include stdLib nodes """
    for node_object in static_analysis_nodes.values():
        #Compute the standard libaray edges to remove, and the new application
        #nodes to replace them with
        new_edges_to_be_added = []
        edges_to_std_lib_to_be_removed = set()
        for edge in node_object.edges:
            #Remove standard library edges
            if edge.dest not in application_method_list:
                edges_to_std_lib_to_be_removed.add(edge)
            #If the destination is in std_lib_nodes, replace with the set of
            #reachable application nodes
            if edge.dest in std_lib_nodes:
                for node in std_lib_nodes[edge.dest].reachable_app_nodes:
                    new_edges_to_be_added.append(Edge(edge.bytecodeOffset, node))
        #Remove the standard library edges
        for edge_name in edges_to_std_lib_to_be_removed:
            node_object.edges.remove(edge_name)

        #Replace the standard library edges with ones to application nodes
        node_object.edges.extend(new_edges_to_be_added)

def remove_stdlib_edges(analysisfile, methodsfile, do_transitive_closure):
    """ Remove edges to the std library, but keep the transitive closures."""
    #Intialize some variables
    nodes = defaultdict(empty_node)
    #Adjacency list representation. However this does not include nodes with no
    #outgoing edges
    appmethodlist = set() #Needed to remove java standard library nodes
    appmethodlist.add(ROOT_METHOD)
    #Read the analysis reachable edges
    with open(analysisfile) as filep:
        #Loop through the edges
        for edge in csv.DictReader(filep):
            #Create new node if it doesn't exist
            _method = edge["method"]
            _offset = edge["offset"]
            _target = edge["target"]
            nodes[_method].edges.append(Edge(_offset, _target))

    #Read the nethod list from the methods.csv file
    with open(methodsfile) as filep:
        lines = [line.rstrip() for line in filep]
        for line in lines:
            appmethodlist.add(line)

    stdlibnodes = get_std_lib_nodes(nodes, appmethodlist)
    remove_stdlib_nodes(nodes, stdlibnodes)
    if do_transitive_closure:
        compute_reachable_application_nodes(stdlibnodes, appmethodlist, nodes)
        replace_std_lib_edges_with_app_edges(nodes, stdlibnodes, appmethodlist)

    return nodes

if __name__ == '__main__':
    main()