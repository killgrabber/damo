#!/bin/sh

cmake -S . -B build
cd build
make

echo "Copying exe..."
