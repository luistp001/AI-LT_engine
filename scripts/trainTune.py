
import sys, os, glob
from multiprocessing import Pool
from mainModel import *
from evaluation import *
from load_util import *
   
#This script test different values for the parameters of the engine.

def evaluateParametersInd( params ):
  modelTemp = mainModel( param1 = params[0], param2 = params[1], param3 = params[2] )   
  return objectEv.evaluate( modelTemp  )

def get_best_values( listParams , scale):
  #Return the best combination of the current list of parameters.
  global numCores
  pool = Pool( processes = numCores )
  listResults  = zip ( *( pool.map( evaluateParametersInd, listParams ) ) )
  listKappas   = list( listResults[0])
  highestKappa = max ( listKappas )
  iHighKappa   = listKappas.index( highestKappa ) #index of highest kappa
  pool.close()
  pool.join()
  if scale == 2:
    return highestKappa, listResults[1][iHighKappa], listResults[2][iHighKappa], listResults[3][iHighKappa], listResults[4][iHighKappa]
  elif scale == 3:
    return highestKappa, listResults[1][iHighKappa], listResults[2][iHighKappa], listResults[3][iHighKappa], listResults[4][iHighKappa], listResults[5][iHighKappa]
  elif scale == 4:
    return highestKappa, listResults[1][iHighKappa], listResults[2][iHighKappa], listResults[3][iHighKappa], listResults[4][iHighKappa], listResults[5][iHighKappa], listResults[6][iHighKappa]
  else:
    return highestKappa, listResults[1][iHighKappa], listResults[2][iHighKappa], listResults[3][iHighKappa]
   
def evaluateParams( currentSet, ngramMax , kInter,  scale, param3List, modelType , level):
  text, originalText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter )
  XTrain, XTest, indices_Train, indices_Test = get_XTrain_XTest( text , y ) #Split Data into trainining and test data

  global objectEv
  #Initialize object to evaluate parameters
  objectEv = evaluation( XTrain = XTrain, XTest = XTest, indices_Train = indices_Train, indices_Test = indices_Test,
                      y = y, y1 = y1, y2 = y2 , scale = scale , modelType = modelType , level = level)
  
  #Different values for testing. "level" paramters control the
  #the number of values to be tested.
  posVal1LG  = [ 3**(i/2.0) for i in range(-5, 6, level) ]
  posVal2LG  = [ i / 10.0   for i in range( 6,14, level) ]
  posVal2GB  = [ 2**(i/2.0) for i in range(-7, 1, level) ]
  posVal1SV1 = [ 3**(i/2.0) for i in range(-1, 6, level) ]
  posVal1SV2 = [ 3**(i/2.0) for i in range(-7,-1, level) ]

  #human kappa
  hKappa  = skllMetrics.kappa( y1, y2 )
  hKappa1 = skllMetrics.kappa( y , y1 )
  hKappa2 = skllMetrics.kappa( y , y2 )
  
  print ngramMax, kInter
  
  for param3 in param3List :
    if   modelType == "LR": posVal1, posVal2 = posVal1LG , posVal2LG if param3 > 0 else [ "NA" ]
    elif modelType == "GB": posVal1, posVal2 = [ 1500 ]  , posVal2GB
    elif modelType == "SV": posVal1, posVal2 = posVal1SV1, posVal1SV2

    listParams = [ ( param1, param2, param3 ) for param1 in posVal1 for param2 in posVal2 ]
    if   scale == 2:
      currentKappa, param1, param2, threshold1, threshold2 = get_best_values ( listParams = listParams , scale=scale )
      print ngramMax, kInter, param3, param1, param2 , threshold1, threshold2, currentKappa, hKappa, hKappa1, hKappa2

    elif scale == 1:
      currentKappa, param1, param2, threshold = get_best_values ( listParams = listParams , scale = scale)
      print ngramMax, kInter,  param3, param1, param2 , threshold, currentKappa, hKappa, hKappa1, hKappa2

    elif scale == 3:
      currentKappa, param1, param2, threshold1, threshold2, threshold3 = get_best_values ( listParams = listParams , scale = scale )
      print ngramMax, kInter, param3, param1, param2 , threshold1, threshold2, threshold3, currentKappa

    elif scale == 4:
      currentKappa, param1, param2, threshold1, threshold2, threshold3, threshold4 = get_best_values ( listParams = listParams , scale = scale )
      print ngramMax, kInter, param3, param1, param2 , threshold1, threshold2, threshold3, threshold4, currentKappa
  return

def evaluateSet( currentSet , scale, ngramList, kInterList, param3List, modelType, level):
  #Evaluate the engine for the current set with different set of parameters. The parameters will be saved
  #and the performance obtained by the engine with each combination are saved.
  print "Training began for set %s" % ( currentSet )
  startTime = time.time()
  for ngram in ngramList:
    for kInter in kInterList:
      if ngram == 1 and kInter > 1: 
        continue    
      evaluateParams( currentSet = currentSet, ngramMax = ngram , kInter = kInter, 
                          scale = scale, param3List = param3List , modelType = modelType, level = level)
  print "Training lasted %f minutes" % ( (time.time() - startTime) / 60 )

if __name__=="__main__":

  modelType = sys.argv[1]
  level    = int( sys.argv[2] ) 
  global numCores
  numCores = int( sys.argv[3] )


  ngramMin, kInterMin, param3Min = 1, 1, 0
  ngramMax, kInterMax  = 5 - level, 5 - level
  if modelType == "LR":
    param3Max = 15
  elif modelType == "GB":
    param3Max = 5 - level
  elif modelType == "SV":
    param3Max = 0
  
  #Values to be tested for the parameters
  kInterList = list( range( kInterMin, kInterMax + 1 ) )
  ngramList  = list( range( ngramMin , ngramMax  + 1 ) )
  param3List = list( range( param3Min, param3Max + 1 ) )
  
  listOfFiles = glob.glob("textTrain/*.tsv")
  listOfFiles.sort()

  for fileName in listOfFiles[0]:
    currentSet = fileName.split("/")[-1][:-4]
    scale = rangeY(currentSet) #Read the maximum score for the current set
    if scale > 2 and modelType != "GB":
      continue
    print "\n\nset %s\twith scale %d\n" %( currentSet, scale)
          
    evaluateSet( currentSet, scale, ngramList, kInterList, param3List, modelType, level)
    
