#!/usr/bin/env bash

find $1 -name $2 -print0 | xargs -0 sed -i "" "s/EDGE_DEPTH_DEFAULT/1000000000/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/INCOMING_EDGES_DEFAULT/-1/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/OUTGOING_EDGES_DEFAULT/-1/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/FANOUT_DEFAULT/1000000000/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/SOURCE_NODE_DEG_DEFAULT/1000000000/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/REACHABLE_FROM_MAIN_DEFAULT/0/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/NUM_PATHS_TO_THIS_DEFAULT/0/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/REPEATED_EDGES_DEFAULT/0/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/DISJOINT_PATHS_DEFAULT/-1/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/FANOUT_AVG_DEFAULT/1000000000/g"
find $1 -name $2 -print0 | xargs -0 sed -i "" "s/FANOUT_MIN_DEFAULT/1000000000/g"