#!/bin/sh

set -o errexit # exit on first error
set -o nounset # unset variables are errors

if [ ${1:-foo} = "--clear-cache" ]; then
    sudo rm -r ~/.cpm-source-cache
fi

rm -rf build

#Eigen wants to be build seperate


cmake -S . -B build
cd build
make
