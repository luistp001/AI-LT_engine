import sys, os, glob, pickle
from multiprocessing import Pool
from mainModel import *
from evaluation import *
from load_util import *
 
#This script trains the engine with the already saved parameters

def trainFinal( currentSet , scale ):
  paramsDict = getParams( currentSet ) #Loads parameters

  if scale <= 2:    
    #Train LR model with loaded parameters
    modelType = "LR"
    ngramMax, kInter, param3, param1, param2 = paramsDict[modelType]  
    text, originalText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter ) 
    modelTLG = tfidf( analyzer = 'word', ngram_range = (1,1) , token_pattern = r'[^ ]+', min_df = 2 ,
                  norm = None, use_idf = False, smooth_idf = False, sublinear_tf = True )
    modelTLG.fit( text )
    X = modelTLG.transform( text )
    modelLG = mainModel( param1 = param1, param2 = param2, param3 = param3  ) 
    if scale > 1:
      modelLG.fit( X, y , "LR2" )
    else:
      modelLG.fit( X, y , "LR1" ) 
    
    #Train SV model with loaded parameters
    modelType = "SV"
    ngramMax, kInter, param3, param1, param2 = paramsDict[modelType]  
    text, originalText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter )
    modelTSV = tfidf( analyzer = 'word', ngram_range = (1,1) , token_pattern = r'[^ ]+', min_df = 2 ,
                  norm = None, use_idf = False, smooth_idf = False, sublinear_tf = True )
    modelTSV.fit( text )
    X = modelTSV.transform( text )  
    modelSV = mainModel( param1 = param1, param2 = param2, param3 = param3  ) 
    if scale > 1:
      modelSV.fit( X, y , "SV2" )
    else:
      modelSV.fit( X, y , "SV1" )
  else:
    modelTLG, modelLG, modelTSV, modelSV = None, None, None, None
  
  #Train GB model with loaded parameters
  modelType = "GB"
  ngramMax, kInter, param3, param1, param2 = paramsDict[modelType]  
  text, originalText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter )
  modelTGB = tfidf( analyzer = 'word', ngram_range = (1,1) , token_pattern = r'[^ ]+', min_df = 2 ,
                  norm = None, use_idf = False, smooth_idf = False, sublinear_tf = True )
  modelTGB.fit( text )
  X = modelTGB.transform( text )  
  modelGB = mainModel( param1 = param1+1, param2 = param2, param3 = param3 ) 
  if scale > 1:
    modelGB.fit( X, y , "GB2" )
  else:
    modelGB.fit( X, y , "GB1" )
  completeModel = [ modelTLG, modelLG, modelTSV, modelSV, modelTGB, modelGB ]
  #Save all models
  pickle.dump(completeModel, open( "savedModels/" + currentSet + ".pickle" , "w"))


def prepTrain ( fileName ):
  currentSet = fileName.split("/")[-1][:-4]
  scale = rangeY(currentSet)
  print "Training set %s\twith scale %d" %( currentSet, scale)    
  trainFinal( currentSet, scale )


if __name__=="__main__":
  numCores = int( sys.argv[1] )
  
  listOfAllFiles = glob.glob("tempTextTrain2/*.tsv")
  listOfAllFiles.sort()

  pool = Pool( processes = numCores )
  pool.map( prepTrain, listOfAllFiles )
  pool.close()
  pool.join()