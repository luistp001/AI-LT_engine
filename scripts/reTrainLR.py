import sys, os, glob
from multiprocessing import Pool
from evaluation import *
from load_util import *
from mainModel import * 

#This script validates the parameters obtained for the Logistic Regression (LR) model.
#It trains the LR model with each combination of parameters, but the model is validated
#using the average of 14 CV cross validations. Each CV cross validation is trained with
#a different version of the data obtained by shuffling the original data.


def evaluateParametersInd( params ):
  modelTemp = mainModel( param1 = params[0], param2 = params[1], param3 = params[2] )   
  return objectEv.evaluate( modelTemp  )

def evalOnly( currentSet , scale , params , modelType, shuffle = 0, KFold = False ):
  ngramMax = params["ngramMax"]
  kInter  = params["kInter"]

  text, oText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter , shuffle) #Get the data after being shuffled
                                                                              #"shuffle" times
  XTrain, XTest, indices_Train, indices_Test = get_XTrain_XTest( text , y )  
  global objectEv
  objectEv = evaluation( XTrain = XTrain, XTest = XTest, indices_Train = indices_Train, indices_Test = indices_Test,
                      y = y, y1 = y1, y2 = y2 , scale = scale , modelType = modelType )
      
  cParams = ( params["param1"], params["param2"], params["nComp"] , KFold )

  cKappa = evaluateParametersInd( cParams )[0]
  return cKappa
   

def evaluateTemp( args ):
  return evalOnly ( args[0], args[1], args[2], args[3], args[4], True)

if __name__=="__main__":

  numCores = int( sys.argv[1] )  
  listOfFiles = glob.glob("textTrain/*.tsv")
  listOfFiles.sort()

  modelType = "LR"  
  for fileName in listOfFiles:
    currentSet = fileName.split("/")[-1][:-4]
    scale = rangeY(currentSet)
    if scale > 2:
      continue
          
    paramsLines = open( "modelResults/LR_topk/%s.txt" % currentSet,'r' ) #Load parameters
    outputFile = open( "modelResults/selected/%s.txt" % currentSet , 'w')
    listParams = list()
    for line in paramsLines:
      
      lineC = line.strip().split(' ')[0:5]
      ngramMax = int(lineC[0])
      kInter   = int(lineC[1])
      nComp    = int(lineC[2])
      param1   = float(lineC[3])
      param2   = float( lineC[4] ) if nComp > 0 else lineC[4][1:-1]
      #Create a list of parameters
      listParams.append( { "ngramMax" : ngramMax, "kInter" : kInter, "nComp": nComp, "param1" : param1, "param2" : param2 } )

    bestParams = None
    highestKappa = 0
    for params in listParams:
      kRep = 14
      cval = 0.0
      largs = [ (currentSet, scale, params, modelType, k+1 ) for k in range(kRep) ]
      #Create a list of arguments for the validation. The additional parameter k is the number of times that the data will be shuffled.
      evaluateTemp ( largs[0] )

      pool = Pool( processes = numCores )      
      currentKappa =  sum( pool.map(  evaluateTemp, largs  ) ) / kRep #Validate the LR model with each combination of arguments
                                                                      #and find the average
      if currentKappa > highestKappa: #Update the parameters
        highestKappa, bestParams = currentKappa, params 
      pool.close()
      pool.join()
    #Print the combination of parameters for the LR model
    outputFile.write( "LR:%d %d %d %g %s %g\n" % ( bestParams["ngramMax"], bestParams["kInter"],  bestParams["nComp"], bestParams["param1"], str(bestParams["param2"] ), highestKappa) )
    outputFile.close()

    


  



