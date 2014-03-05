#!/bin/bash

#This script calculates the score for each dataset using the trained engine

numCores=4

stage=Test
mkdir -p tempText$stage
mkdir -p tempText${stage}2
python scripts/preProcess.py $stage $numCores

mkdir -p testPredictions
python scripts/testFinal.py $numCores


