#!/bin/bash

#This script trains the engine with the already saved hyper-parameters.

numCores=4

stage=Train
mkdir -p tempText$stage
mkdir -p tempText${stage}2
python scripts/preProcess.py $stage $numCores

mkdir -p savedModels
python scripts/trainFinal.py $numCores

