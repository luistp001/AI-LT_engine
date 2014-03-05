import time, copy, skllMetrics
import pandas as pd
import numpy as np

#This file contains all the functions that evaluate and validate the models

class evaluation():
  
  def __init__(self, XTrain, XTest, indices_Train, indices_Test,
                     y, y1, y2 , scale, modelType, cv = 5, level = 1):
    self.y = y 
    self.scale = scale
    self.cv = cv 
    self.indices_Train = indices_Train 
    self.indices_Test = indices_Test    
    self.XTrain = XTrain
    self.XTest = XTest
    self.modelType = modelType
    self.level = level

  def evaluate( self, model ):
    if self.modelType == "GB" and self.scale == 2:   
      return self.evaluateGB2( model , model.param1 )

    if self.modelType == "GB" and self.scale == 1:   
      return self.evaluateGB1( model , model.param1 )
    
    if self.modelType == "GB" and self.scale > 2:
      return self.evaluateGB3( model , model.param1 , scale = self.scale )

    if self.scale == 1:
      return self.evaluateCl1( model , SVM = self.modelType == "SV")

    if self.scale == 2:
      return self.evaluateCl2( model , SVM = self.modelType == "SV")

  def evaluateGB1(self, model, nTree, fixed = False):
    #This function evaluates the Gradient Boosting Machine Model when the possible 
    #scores may be 0 or 1. It returns the tree and threshold that produce the highest kappa.

    yProbL = self.cv_estimateGB1( model , nTree)
    bestKappa, bthreshold, bTree = 0, 0, 0
    NE = 20
    treeCandidates = [nTree-1] if fixed else [ 20, 30, 50, 100, 150, 200, 250, 300, 350 , 400, 450, 500 , 550, 600, 750, 1000, 1250, 1500 ]
    for j in treeCandidates:
      cTree= j - 1
      yProb = yProbL[cTree]
      for div in range(1,20):   
        threshold = div / 100.0
        yPred = yProb > threshold
        ckappa = skllMetrics.kappa( self.y, yPred )
        if ckappa > bestKappa:          
          bestKappa, bthreshold, bTree = ckappa, threshold, cTree 
    return bestKappa, bTree, model.param2, bthreshold

  def evaluateGB2(self, model, nTree, fixed = False):
    #This function evaluates the Gradient Boosting Machine Model when the possible 
    #scores from 0 to 2. It returns the tree and thresholds that produce the highest kappa.    

    yPredRL = self.cv_estimateGB2( model , nTree)
    bestKappa, bthreshold1, bthreshold2, bTree = 0, 0, 0, 0
    NE = 20
    treeCandidates = [nTree-1] if fixed else [ 20, 30, 50, 100, 150, 200, 250, 300, 350 , 400, 450, 500 , 550, 600, 750, 1000 ]
    for j in treeCandidates:
      cTree = j - 1
      if cTree >= len( yPredRL ) :
        break
      cKappa, threshold1, threshold2 = self.get_best_kappa2( NE, yPredRL[cTree] )
      if cKappa > bestKappa:
        bestKappa, bthreshold1, bthreshold2, bTree = cKappa, threshold1, threshold2, cTree 
    return bestKappa, bTree, model.param2, bthreshold1, bthreshold2

  def evaluateGB3(self, model, nTree, fixed = False, scale = 3):
    #This function evaluates the Gradient Boosting Machine Model when the possible 
    #scores from 0 to 3 or 4. It returns the tree and thresholds that produce the highest kappa.    
    if not fixed:
      nTree = nTree / ( scale ** 2 )
    yPredRL = self.cv_estimateGB2( model , nTree)
    bestKappa, bTree = 0, 0
    candidates = [nTree-1] if fixed else [ 20, 30, 50, 100, 150, 200, 250, 300, 350 , 400, 450, 500 , 550, 600, 750, 1000 ]
    for j in candidates:
      cTree = j - 1
      if ( cTree >= len( yPredRL) ):
        continue            
      if scale == 3:
        NE = 8 - self.level
        cKappa, thresholds = self.get_best_kappa3( NE, yPredRL[cTree] )
      elif scale == 4:        
        NE = 6 - self.level
        cKappa, thresholds = self.get_best_kappa4( NE, yPredRL[cTree] )
      if cKappa > bestKappa:
        bestKappa, bthresholds, bTree = cKappa, thresholds, cTree 
    if scale == 3:
      return bestKappa, bTree, model.param2, thresholds[0], thresholds[1], thresholds[2]
    elif scale == 4:
      return bestKappa, bTree, model.param2, thresholds[0], thresholds[1], thresholds[2], thresholds[3]

  def evaluateCl1( self, model  , SVM = False):
    #This function evaluates Logistic Regression or Support Vector Machine Model when the possible 
    #scores may be 0 or 1. It returns the threshold and current parameters that produce the highest kappa.
    yProb = self.cv_estimateCl1( model, SVM )
    bestKappa, bthreshold = 0, 0.5
    for div in range(1,100): 
      threshold = div / 100.0
      yPred = yProb > threshold
      ckappa = skllMetrics.kappa( self.y, yPred )
      if ckappa > bestKappa:
        bestKappa = ckappa
        bthreshold = threshold 
    return bestKappa, model.param1, model.param2, bthreshold

  def evaluateCl2( self, model , SVM = False):  
    #This function evaluates Logistic Regression or Support Vector Machine Model when the possible 
    #scores may vary from 0 to 2. It returns the threshold and current parameters that produce the highest kappa.
    NE = 20
    if SVM:
      bestKappa, bthreshold1, bthreshold2 = self.evaluateSV2( model )
      return bestKappa, model.param1, model.param2, bthreshold1, bthreshold2
    yProb1, yProb2 = self.cv_estimateCl2( model , SVM)    
    bestKappa, bthreshold1, bthreshold2 = 0, 0.5, 0.5
    for thresholdV1 in range(1,NE): 
      for thresholdV2 in range(1,NE): 
        threshold1 = thresholdV1 / ( NE * 1.0 )
        threshold2 = thresholdV2 / ( NE * 1.0 )
        yPred1 = ( yProb1 > threshold1 ) * 1 
        yPred2 = ( yProb2 > threshold2 ) * 2
        yPred = np.maximum( yPred1, yPred2)
        ckappa = skllMetrics.kappa( self.y, yPred )
        if ckappa > bestKappa:
          bestKappa, bthreshold1, bthreshold2 = ckappa, threshold1, threshold2
    return bestKappa, model.param1, model.param2, bthreshold1, bthreshold2

  def evaluateSV2(self, model ):
    #This function evaluates Support Vector Machine Model when the possible 
    #scores may vary from 0 to 2. It returns the threshold and current parameters that produce the highest kappa.
    yPredR = self.cv_estimateSV2( model )
    bestKappa, bthreshold1, bthreshold2 = 0, 0, 0
    NE = 20    
    return self.get_best_kappa2( NE, yPredR )

  def get_best_kappa2( self, NE, yPredR ):
    #This function returns the thresholds that produce the highest kappa
    #when the score can vary from 0 to 2
    bestKappa, bthreshold1, bthreshold2 = 0, 0, 0
    yPred = yPredR
    for thresholdV1 in range( 1, 2 * NE ):       
      for thresholdV2 in range( thresholdV1 + 1 , 2 * NE):
        threshold1  = thresholdV1 / ( NE * 1.0 )
        threshold2  = thresholdV2 / ( NE * 1.0 )
        if threshold1 < np.min( yPredR ): 
          continue
        if threshold2 > np.max( yPredR ): 
          continue    
        yPred = ( yPredR >= threshold1 ) * 1        
        yPred = ( yPredR >= threshold2 ) * 1 + yPred
        cKappa = skllMetrics.kappa( self.y, yPred )
        if cKappa > bestKappa:
          bestKappa, bthreshold1, bthreshold2 = cKappa, threshold1, threshold2
    return bestKappa, bthreshold1, bthreshold2

  def get_best_kappa3( self, NE, yPredR ):
    #This function returns the thresholds that produce the highest kappa
    #when the score can vary from 0 to 3
    bestKappa, bthreshold1, bthreshold2, bthreshold3 = 0, 0, 0, 0
    yPred = yPredR
    
    for thresholdV1 in range( 1, 3 * NE ):       
     for thresholdV2 in range( thresholdV1 + 1 , 3 * NE):
      for thresholdV3 in range( thresholdV2 + 1 , 3 * NE):
       threshold1  = thresholdV1 / ( NE * 1.0 )
       threshold2  = thresholdV2 / ( NE * 1.0 )
       threshold3  = thresholdV3 / ( NE * 1.0 )
       if threshold1 < np.min( yPredR ): 
         continue
       if min( [ threshold2, threshold3 ] ) > np.max( yPredR ) : 
         continue    
       yPred = ( yPredR >= threshold1 ) * 1        
       yPred = ( yPredR >= threshold2 ) * 1 + yPred
       yPred = ( yPredR >= threshold3 ) * 1 + yPred
       cKappa = skllMetrics.kappa( self.y, yPred )
       if cKappa > bestKappa:
        bestKappa, bthreshold1, bthreshold2, bthreshold3 = cKappa, threshold1, threshold2, threshold3
    return bestKappa, [ bthreshold1, bthreshold2, bthreshold3 ]

  def get_best_kappa4( self, NE, yPredR ):
    #This function returns the thresholds that produce the highest kappa
    #when the score can vary from 0 to 4
    bestKappa, bthreshold1, bthreshold2, bthreshold3, bthreshold4 = 0, 0, 0, 0, 0
    yPred = yPredR    
    for thresholdV1 in range( 1, 2 * NE ):       
      for thresholdV2 in range( thresholdV1 + 1 , 2 * NE):
        for thresholdV3 in range( thresholdV2 + 1 , 3 * NE):
          for thresholdV4 in range( thresholdV3 + 1 , 4 * NE):
            threshold1  = thresholdV1 / ( NE * 1.0 )
            threshold2  = thresholdV2 / ( NE * 1.0 )
            threshold3  = thresholdV3 / ( NE * 1.0 )
            threshold4  = thresholdV4 / ( NE * 1.0 )
            if threshold1 < np.min( yPredR ): 
              continue
            if min( [ threshold2, threshold3, threshold4 ] ) > np.max( yPredR ) : 
              continue    
            yPred = ( yPredR >= threshold1 ) * 1        
            yPred = ( yPredR >= threshold2 ) * 1 + yPred
            yPred = ( yPredR >= threshold3 ) * 1 + yPred
            yPred = ( yPredR >= threshold4 ) * 1 + yPred
            cKappa = skllMetrics.kappa( self.y, yPred )
            if cKappa > bestKappa:
              bestKappa, bthreshold1, bthreshold2, bthreshold3, bthreshold4 = cKappa, threshold1, threshold2, threshold3, threshold4
    return bestKappa, [ bthreshold1, bthreshold2, bthreshold3, bthreshold4 ]
    
  def cv_estimateGB1( self, model , nTree):
    #This function returns the predictions of the GB model for the training data
    #using cross validation when the possible scores may be from 0 to 1
    yProb = pd.Series ( 0., index = self.y.index)
    yProbL = [ yProb.copy() for i in range( nTree ) ]

    for i in range( self.cv  ):
      iTrain, iTest = self.indices_Train[i], self.indices_Test[i]
      modelCopy = copy.deepcopy( model )  
      modelCopy.fit( self.XTrain[i] , self.y.iloc[ iTrain ], "GB1" )
      for cTree, y_probj in enumerate( modelCopy.staged_decision_function( self.XTest[i] ) ):
        yProbL[cTree].iloc[ iTest ] = y_probj[:,1]
    return yProbL

  def cv_estimateGB2( self, model , nTree):
    #This function returns the predictions of the GB model for the training data
    #using cross validation when the possible scores may vary from 0 to 2
    yPredR = pd.Series ( 0., index = self.y.index)
    yPredRL = [ yPredR.copy() for i in range( nTree ) ]
    for i in range( self.cv  ):
      iTrain, iTest = self.indices_Train[i], self.indices_Test[i]
      modelCopy = copy.deepcopy( model )  
      modelCopy.fit( self.XTrain[i] , self.y.iloc[ iTrain ], "GB2" )
      for cTree, y_predj in enumerate( modelCopy.staged_decision_function( self.XTest[i] ) ):
        if cTree < len( yPredRL ):
          yPredRL[cTree].iloc[ iTest ] = y_predj
    return yPredRL

  def cv_estimateCl1 ( self, model, SVM ):
    #This function returns the predictions of the LR or SV models for the training data
    #using cross validation when the possible scores may vary from 0 to 1
    yProb = pd.Series ( 0., index = self.y.index)
    
    for i in range( self.cv  ):
      iTrain, iTest = self.indices_Train[i], self.indices_Test[i]
      modelCopy = copy.deepcopy( model )  
      modelType = "SV1" if SVM else "LR1"
      modelCopy.fit( self.XTrain[i] , self.y.iloc[ iTrain ] , modelType )
      mprob = modelCopy.predict_proba( self.XTest[i] )
      yProb.iloc[ iTest ]  = np.array( mprob[:,1] )
    return yProb

  def cv_estimateCl2 ( self, model , SVM ):
    #This function returns the predictions of the LR or SV models for the training data
    #using cross validation when the possible scores may vary from 0 to 2
    yProb1 = pd.Series ( 0., index = self.y.index)
    yProb2 = pd.Series ( 0., index = self.y.index)
    
    for i in range( self.cv  ):      
      iTrain, iTest = self.indices_Train[i], self.indices_Test[i]
      modelCopy = copy.deepcopy( model )  
      modelType = "SV2" if SVM else "LR2"
      modelCopy.fit( self.XTrain[i] , self.y.iloc[ iTrain ], modelType )
      yProb1.iloc[ iTest ], yProb2.iloc[ iTest ] = modelCopy.predict_proba( self.XTest[i] )
    return yProb1, yProb2

  def cv_estimateSV2( self, model ):
    #This function returns the predictions of the SV model for the training data
    #using cross validation when the possible scores may vary from 0 to 1
    yPredR = pd.Series ( 0., index = self.y.index)
    
    for i in range( self.cv  ):
      iTrain, iTest = self.indices_Train[i], self.indices_Test[i]
      modelCopy = copy.deepcopy( model )  
      modelCopy.fit( self.XTrain[i] , self.y.iloc[ iTrain ], "SV2" )
      yPredR.iloc[ iTest ] = modelCopy.predict( self.XTest[i] )
    return yPredR





