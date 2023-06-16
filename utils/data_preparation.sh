#!/bin/bash

# filein="data/roadNet-CA/raw/uduw.el"
# directed=0
# weighted=0
filein=$1
directed=$2
weighted=$3

# chekc 3 inputs are given
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <filein> <directed> <weighted>"
    echo "directed: 0 for undirected, 1 for directed"
    echo "weighted: 0 for unweighted, 1 for weighted"
    exit 1
fi

# check if file exists
if [[ ! -f $filein ]]; then
    echo "File $filein does not exist!"
    exit 1
fi

if [[ $directed -eq 1 ]]; then
    if [[ $weighted -eq 1 ]]; then # directed and weighted
        # remove lines start with #
        fileout=${filein}_uncomment
        sed -e '/^#/d' $filein > $fileout
        mv $fileout $filein

        # substitute , with space, for cage14.mtx only
        fileout=$(echo ${filein} | sed 's/[^/]*$/dw.wel/g')
        sed -e 's/,/ /g' $filein > $fileout

        # sort by src then dst
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/dw.wel.sorted/g')
        ./utils/bin/utils -i $filein -o $fileout -m 6

        # remove isolated nodes, also to 0-based
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/dw.wel/g')
        ./utils/bin/utils -i $filein -o $fileout -m 5

        # symmetrize
        fileout=$(echo ${filein} | sed 's/[^/]*$/dw.sym.wel/g')
        ./utils/bin/utils -i $filein -o $fileout -m 11

        # dw2udw
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/udw.wel/g')
        ./utils/bin/utils -i $filein -o $fileout -m 3

    else # directed and unweighted
        # remove lines start with #
        fileout=${filein}_uncomment
        sed -e '/^#/d' $filein > $fileout
        mv $fileout $filein

        # substitute \t with space
        fileout=$(echo ${filein} | sed 's/[^/]*$/duw.el/g')
        sed -e 's/\t/ /g' $filein > $fileout

        # sort by src then dst
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/duw.el.sorted/g')
        ./utils/bin/utils -i $filein -o $fileout -m 6

        # remove isolated nodes, also to 0-based
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/duw.el/g')
        ./utils/bin/utils -i $filein -o $fileout -m 5

        # symmetrize
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/duw.sym.el/g')
        ./utils/bin/utils -i $filein -o $fileout -m 11

        # duw2uduw
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/uduw.el/g')
        ./utils/bin/utils -i $filein -o $fileout -m 1
    fi
else
    if [[ $weighted -eq 1 ]]; then # undirected and weighted
        # remove lines start with #
        fileout=${filein}_uncomment
        sed -e '/^#/d' $filein > $fileout
        mv $fileout $filein

        # substitute \t with space
        fileout=$(echo ${filein} | sed 's/[^/]*$/udw.wel/g')
        sed -e 's/\t/ /g' $filein > $fileout

        # sort by src then dst
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/udw.wel.sorted/g')
        ./utils/bin/utils -i $filein -o $fileout -m 6

        # remove isolated nodes, also to 0-based
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/udw.wel/g')
        ./utils/bin/utils -i $filein -o $fileout -m 5

        # udw2dw
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/dw.wel/g')
        ./utils/bin/utils -i $filein -o $fileout -m 4
    else # undirected and unweighted
        # remove lines start with #
        fileout=${filein}_uncomment
        sed -e '/^#/d' $filein > $fileout
        mv $fileout $filein

        # substitute \t with space
        fileout=$(echo ${filein} | sed 's/[^/]*$/uduw.el/g')
        sed -e 's/\t/ /g' $filein > $fileout

        # sort by src then dst
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/uduw.el.sorted/g')
        ./utils/bin/utils -i $filein -o $fileout -m 6

        # remove isolated nodes, also to 0-based
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/uduw.el/g')
        ./utils/bin/utils -i $filein -o $fileout -m 5

        # uduw2duw
        filein=$fileout
        fileout=$(echo ${filein} | sed 's/[^/]*$/duw.el/g')
        ./utils/bin/utils -i $filein -o $fileout -m 2
    fi
fi



