#!/bin/sh

set -o errexit # exit on first error
set -o nounset # unset variables are errors

# Build latex files ...
echo "Building latex files"
cd src/ausarbeitung
pdflatex main.tex
cd ..
cd ..
cp src/ausarbeitung/main.pdf ausarbeitung.pdf

echo "latex files finished..."

if [ ${1:-foo} = "--clear-cache" ]; then
    sudo rm -r ~/.cpm-source-cache
fi

rm -rf build

#Eigen wants to be build seperate


cmake -S . -B build
cd build
make

