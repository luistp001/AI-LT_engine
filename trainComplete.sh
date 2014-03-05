#!/bin/bash

#This script trains all the hyper-parameters of the engine
#numCores is the number of cores available for training in parallel
#level can take the values of 1, 2 or 3. This parameter indicates the number 
#of values to be tested for the parameters of the engine. A value of 1 will
#test the most number of values, so the performance of the engine will be
#higher but it will take much longer to run. A value of 3 will test few
#parameters, so the performance will be lower but it will much faster to train.
#The difference of performances may be slight such as 0.03 of difference in 
#quadratic kappa.

numCores=4
level=3

stage=Train
mkdir -p tempText$stage
mkdir -p tempText${stage}2
python scripts/preProcess.py $stage $numCores

mkdir -p modelOutput
for modelType in  LR SV GB
do
  echo "Training parameters for ${modelType} model ..."
  mkdir -p modelOutput/${modelType}
  python scripts/trainTune.py ${modelType} $level $numCores > modelOutput/${modelType}/results.txt
  mkdir -p modelResults
  mkdir -p modelResults/${modelType}_topk
  python scripts/orgOutput.py $modelType
done

mkdir -p modelResults/LRupd_topk
mkdir -p modelResults/selected
echo "Training ensemble parameters ..."
python scripts/reTrainLR.py $numCores
python scripts/getBestCandidate.py
python scripts/trainEnsemble.py $level $numCores

