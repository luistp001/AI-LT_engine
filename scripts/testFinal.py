import sys, os, glob, pickle
from multiprocessing import Pool
from mainModel import *
from evaluation import *
from load_util import *

def getPred( currentSet , scale , Params ):
  
  paramsDict = getParams( currentSet = currentSet, scale = scale)   
  completeModel = pickle.load( open( "savedModels/" + currentSet + ".pickle" , "rb"))
  if scale <= 2:
    #Calculates score obtained by LR model
    modelType = "LR"
    ngramMax , kInter = paramsDict[modelType][0:2] 
    text = get_data_test ( currentSet, ngramMax, kInter )
    modelTLG, modelLG = completeModel[0:2]
    X  = modelTLG.transform( text )
    if scale > 1:
      yProb1, yProb2 = modelLG.predict_proba( X )
      yPredLR = np.maximum( yProb1, yProb2 * 2 )
    else:    
      yPredLR = modelLG.predict_proba( X )[:,1]

    #Calculates score obtained by SV model
    modelType = "SV"
    ngramMax , kInter = paramsDict[modelType][0:2] 
    text = get_data_test ( currentSet, ngramMax, kInter )
    modelTSV, modelSV = completeModel[2:4]
    X = modelTSV.transform( text )    
    yPredSV = modelSV.predict( X ) if scale > 1 else modelSV.predict_proba( X )[:,1]
  
  #Calculates score obtained by GB model
  modelType = "GB"
  ngramMax , kInter = paramsDict[modelType][0:2] 
  param1 =  paramsDict[modelType][3]  
  text = get_data_test ( currentSet, ngramMax, kInter )
  modelTGB, modelGB = completeModel[4:]
  X = modelTGB.transform( text )  
  for j, y_predj in enumerate( modelGB.staged_decision_function( X ) ):
    if j == param1:
      break
  yPredGB = y_predj[:,0] if scale > 1 else y_predj[:,1]
          
  #Combine the the scores of the three models with the saved weights
  #and the saved thresholds
  if scale == 2:
    coef1, coef2, coef3, threshold1, threshold2 = Params
    probF = coef1 * yPredLR + coef2 * yPredSV + coef3 * yPredGB
    yPred = ( probF >= threshold1 ) * 1        
    yPred = ( probF >= threshold2 ) * 1 + yPred
  elif scale == 1:
    coef1, coef2, coef3, threshold = Params
    probF = coef1 * yPredLR + coef2 * yPredSV + coef3 * yPredGB
    yPred = probF > threshold
  else:
    threshold1, threshold2, threshold3 = paramsDict["GB"][5:8]
    yPred   = ( yPredGB >= float(threshold1) ) * 1        
    yPred   = ( yPredGB >= float(threshold2) ) * 1 + yPred
    yPred   = ( yPredGB >= float(threshold3) ) * 1 + yPred
    if scale == 4:
      threshold4 = paramsDict["GB"][8]
      yPred = ( yPredGB >= float(threshold4) ) * 1 + yPred
  #save the scores
  np.savetxt( "testPredictions/" + currentSet + ".csv", yPred, delimiter = ',', fmt= '%.0f')

def loadWgs():
  #Load weights for combining models
  #and thresholds
  paramsFile = open( "finalParams.txt", 'r' )
  parWgs = {}
  for line in paramsFile:
    lineSplit = line.strip().split(":")
    parWgs[ lineSplit[0] ] = lineSplit[1]
  return parWgs

def prepTest ( fileName ):
  #Load parameters for models
  currentSet = fileName.split("/")[-1][:-4]
  scale = rangeY(currentSet)

  print "set %s\twith scale %d" %( currentSet, scale)  
  if scale <= 2:
    cWgs = parWgs[ currentSet ].split(',')
    for i in range( len( cWgs ) ):
      cWgs[i] = float( cWgs[i] )  
    getPred ( currentSet, scale , cWgs[:-1] )  
  else:
    getPred( currentSet, scale, None )
    
if __name__=="__main__":

  numCores = int( sys.argv[1] )

  listOfFiles = glob.glob("textTest/*.tsv")
  listOfFiles.sort()
  
  global parWgs
  parWgs = loadWgs()
  for fileName in listOfFiles:
    prepTest( fileName )
  #pool = Pool( processes = numCores )
  #pool.map( prepTest, listOfFiles )
  #pool.close()
  #pool.join()

