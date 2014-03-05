import sys, os, glob
from multiprocessing import Pool
from mainModel import *
from evaluation import *
from load_util import *

#This script trains the weights to combine the Logistic Regression, Support Vector Machine, and Gradient
#Boosting Machine models. It also trains the thresholds to convert the real value predictions to integer scores.
 
def trainFull( currentSet , scale, listParams, shuffle = 0  ):
  
  shuffle += 20
  paramsDict = getParams( currentSet ) #Load parameters

  #Train LR model with loaded parameters
  modelType = "LR" 
  ngramMax, kInter, param3, param1, param2 = paramsDict[modelType]  
  text, originalText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter , shuffle)
  XTrain, XTest, indices_Train, indices_Test = get_XTrain_XTest( text , y )  
  global objectEv
  objectEv = evaluation( XTrain = XTrain, XTest = XTest, indices_Train = indices_Train, indices_Test = indices_Test,
                      y = y, y1 = y1, y2 = y2 , scale = scale , modelType = modelType )       
  modelLG = mainModel( param1 = param1, param2 = param2, param3 = param3 ) 
  if scale > 1:
    yProb1, yProb2 = objectEv.cv_estimateCl2 ( modelLG , False)
    yp1 = np.maximum( yProb1, yProb2 * 2 )
  else:
    yp1 = objectEv.cv_estimateCl1( modelLG, False )

  #Train SV model with loaded parameters
  modelType = "SV"
  ngramMax, kInter, param3, param1, param2 = paramsDict[modelType]
  text, originalText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter , shuffle)
  XTrain, XTest, indices_Train, indices_Test = get_XTrain_XTest( text , y )  
  objectEv = evaluation( XTrain = XTrain, XTest = XTest, indices_Train = indices_Train, indices_Test = indices_Test,
                      y = y, y1 = y1, y2 = y2 , scale = scale , modelType = modelType )       
  modelSV = mainModel( param1 = param1, param2 = param2, param3 = param3 ) 
  if scale > 1:
    yp2 = objectEv.cv_estimateSV2 ( modelSV )
  else:
    yp2 = objectEv.cv_estimateCl1( modelSV, True )
  
  #Train GB model with loaded parameters
  modelType = "GB"
  ngramMax, kInter, param3, param1, param2 = paramsDict[modelType]    
  text, originalText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter , shuffle)
  XTrain, XTest, indices_Train, indices_Test = get_XTrain_XTest( text , y )
  objectEv = evaluation( XTrain = XTrain, XTest = XTest, indices_Train = indices_Train, indices_Test = indices_Test,
                      y = y, y1 = y1, y2 = y2 , scale = scale , modelType = modelType )       
  modelGB = mainModel( param1 = param1+1, param2 = param2, param3 = param3 )   
  if scale > 1:
    yL = objectEv.cv_estimateGB2( modelGB , param1 + 1)
  else:
    yL = objectEv.cv_estimateGB1( modelGB , param1 + 1)
  yp3 = yL[ param1 ]
  
  #Calculates the kappas obtained by using the weights and thresholds 
  #in the list of parameters
  prevCoef1 = -1
  prevCoef2 = -1
  prevCoef3 = -1
  listKappas = list()
  for cParams in listParams:
    if scale > 1:
      coef1, coef2, coef3, threshold1, threshold2 = cParams
      if  prevCoef1 != coef1 or prevCoef2 != coef2 or prevCoef3 != coef3:
        yF = coef1 * yp1 + coef2 * yp2 + coef3 * yp3
      yPred = ( yF >= threshold1 ) * 1        
      yPred = ( yF >= threshold2 ) * 1 + yPred
    else:
      coef1, coef2, coef3, threshold = cParams
      if  prevCoef1 != coef1 or prevCoef2 != coef2 or prevCoef3 != coef3:
        yF = coef1 * yp1 + coef2 * yp2 + coef3 * yp3
      yPred = yF > threshold
    ckappa = skllMetrics.kappa( objectEv.y, yPred )
    listKappas.append( ckappa )
    prevCoef1, prevCoef2, prevCoef3 = coef1, coef2, coef3
  
  return listKappas

def trainIWg( args ):
  return trainFull( args[0], args[1], args[2], args[3] )

def trainWgs( currentSet , scale  ):

  listParams = list()
  global level
  ndiv = 20 / level
  thresholdDiv2 = 20 / level
  thresholdDiv1 = 100 / level

  #The following nested for loops produce a list with different values for the weights and thresholds
  for div1 in range(ndiv+1):
    for div2 in range(ndiv+1-div1):
      div3 = ndiv - div1 - div2 
      coef1, coef2, coef3 = float(div1) / ndiv, float(div2) / ndiv , float(div3) / ndiv 
      if scale > 1:
        for thresholdV1 in range( 1, 2 * thresholdDiv2 ):       
          for thresholdV2 in range( thresholdV1 + 1 , 2 * thresholdDiv2):
            threshold1  = float(thresholdV1) / thresholdDiv2 
            threshold2  = float(thresholdV2) / thresholdDiv2 
            listParams.append( ( coef1, coef2, coef3, threshold1, threshold2 ) )
      else:
        for thresholdV in range(thresholdDiv1): 
          threshold = float(thresholdV) / thresholdDiv1
          listParams.append( ( coef1, coef2, coef3, threshold ) )

  nRep = 16
  #Each combination of values is tested nRep times with a different version of the 
  #training data obtained after shuffling the original data
  listArgs = [ ( currentSet, scale, listParams, shuffle) for shuffle in range(nRep) ]
  global numCores
  pool = Pool( processes = numCores )  
  matrixKappas = pool.map( trainIWg, listArgs)
  pool.close()
  pool.join()
  
  highestKappa, bestI = 0.0, 0
  for i in range( len( listParams) ):
    ckappa = 0
    for j in range(len(matrixKappas)): #These three lines average the kappas
      ckappa += matrixKappas[j][i]     #obtained with the ith combination
    ckappa = ckappa / float( nRep )    #of parameters
    if ckappa > highestKappa: #Update parameters if needed
      highestKappa, bestI = ckappa, i
  #Return the combination of parameters that produces the highest kappa
  return [ str(val) for val in listParams[bestI] ], highestKappa

if __name__=="__main__":

  global level
  global numCores
  level, numCores = int(sys.argv[1]), int( sys.argv[2] )

  listOfFiles = glob.glob("textTrain/*.tsv")
  listOfFiles.sort()

  outputFile = open( "finalParams.txt", 'w')
  for fileName in listOfFiles:
    currentSet = fileName.split("/")[-1][:-4]
    scale = rangeY(currentSet)
    if scale > 2:
      continue
    print "set %s\twith scale %d" %( currentSet, scale)    
    
    params, kappa = trainWgs( currentSet, scale )
    outputFile.write( "%s:%s,%.4f\n" % ( currentSet , ",".join( params ), kappa) )

  outputFile.close()