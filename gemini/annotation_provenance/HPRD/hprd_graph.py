#!/usr/bin/env python

#NOTE: we just made this into an edges text file by dumping the pygraph
#representation for gemini 0.18

import cPickle
from pygraph.classes.graph import graph
from pygraph.classes.exceptions import AdditionError

"""
This script converts a binary protein-protein interaction
file from HPRD (http://http://www.hprd.org/) into a cPickled
graph generated by the python-graph library.  The resulting
file (hprd_interaction_graph) can be subsequently explored
by gemini for identifying interacting (direct or indirect) genes
with certain classes of genetic variation in one or more samples.

Input:  BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt
    - from: http://www.hprd.org/download/HPRD_Release9_041310.tar.gz
Output: hprd_interaction_graph

"""

gr = graph()
with open('hprd_interaction_graph', 'wb') as output:
    for line in open("BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt", 'r'):
        fields=line.strip().split("\t")
        first = str(fields[0])
        second = str(fields[3])
        if first != "-":
            try:
                gr.add_nodes([first])
            except AdditionError:
                pass;
        if second != "-":
            try:
                gr.add_nodes([second])
            except AdditionError:
                pass;

        if first != "-" and second != "-" and first != second:
            try:
                gr.add_edge((first, second))
            except AdditionError:
                pass;

    cPickle.dump(gr, output)
