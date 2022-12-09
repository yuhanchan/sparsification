#!/bin/bash

# graph_name=random_100000_100000000

# ./bin/main ./test_graphs/${graph_name}.in > ${graph_name}.cpp.out &
# P1=$!

# psrecord $P1 --include-children --interval 1 --plot ${graph_name}.cpp.png &
# P2=$!

# wait $P1 $P2

# python main.py ./test_graphs/${graph_name}.in > ${graph_name}.py.out &
# P3=$!

# psrecord $P3 --include-children --interval 1 --plot ${graph_name}.py.png &
# P4=$!

# wait $P1 $P2 $P3 $P4
# echo "Done"




# ./bin/main /data3/chenyh/sparsification/data/Reddit/raw/duw.el > Reddit.cpp.out &
python main.py > Reddit.py.out &
P1=$!

# psrecord $P1 --include-children --interval 1 --plot Reddit.cpp.png &
psrecord $P1 --include-children --interval 1 --plot Reddit.py.png --log Reddit.py.log &
P2=$!

wait $P1 $P2
echo "Done"
