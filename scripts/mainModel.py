import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import *
from sklearn.svm import SVC
from sklearn.svm import SVR

#This file contains the main class and functions that train the engine

class mainModel():
  
  def __init__(self, param1 = 1, param2 = 1, param3 = 1 ):    
    #Initialize parameters
    self.param1, self.param2, self.param3  = param1, param2, param3

  def fit( self, X, y , modelType ):
    #Fit depending on the type of model
    self.modelType = modelType
    if self.modelType == "GB2":     self.fitGB2 ( X, y )
    if self.modelType == "GB1":     self.fitGB1 ( X, y )
    if self.modelType == "LR1":     self.fitLR1 ( X, y )
    if self.modelType == "LR2":     self.fitLR2 ( X, y )
    if self.modelType == "SV1":     self.fitSV1 ( X, y )
    if self.modelType == "SV2":     self.fitSV2 ( X, y )

  def fitSV1( self, X, y ):
    #Fit Support Vector Machine when the possible scores may be 0 or 1
    self.model01 = SVC(kernel = 'rbf', C = self.param1, gamma = self.param2, probability = True, cache_size = 2048, random_state = 2512 )    
    self.model01.fit( X , y )

  def fitSV2( self, X, y ):
    #Fit Support Vector Machine when the possible scores may vary from 0 to 2
    self.model01 = SVR(kernel = 'rbf', C = self.param1, gamma = self.param2, probability = True, cache_size = 2048, random_state = 2512 )
    self.model01.fit( X , y )

  def fitGB2( self, X, y ):
    #Fit Gradient Boosting Machine when the possible scores may vary from 0 to 2, 3 or 4
    self.model01 = GradientBoostingRegressor( learning_rate = self.param2, n_estimators = self.param1, 
                                  subsample = 1.0, max_depth = self.param3 + 1, random_state = 2512, max_features = "sqrt")
    self.model01.fit( X.toarray(), y )

  def fitGB1( self, X, y ):
    #Fit Gradient Boosting Machine when the possible scores may be 0 or 1
    self.model01 = GradientBoostingClassifier( learning_rate = self.param2, n_estimators = self.param1, 
                                  subsample =  1.0, max_depth = self.param3 + 1, random_state = 2512, max_features = "sqrt")
    self.model01.fit( X.toarray(), y )
  
  def fitLR1( self, X, y):  
    #Fit Logistic Regression when the possible scores may be 0 or 1
    C1, C2, self.nComp = self.param1, self.param2, self.param3
    #Fit Initial Logistic Regression
    self.model01 = LogisticRegression( penalty= 'l1' , C = C1, random_state=2512 )
    self.model01.fit( X , y )
    #Select features with positive weights
    self.featSel = [ i for i in range(len( self.model01.coef_[0])) if self.model01.coef_[0][i] > 0 ]
    if self.nComp == 0 or len( self.featSel) <= self.nComp:
      return
    
    newX = X[ :, self.featSel ]
    #Train Singular Value decomposition with selected features
    self.concepts = TruncatedSVD( n_components = self.nComp, random_state = 2512 )
    self.concepts.fit( newX )
    #Transform data to selected concepts.
    X1  = self.concepts.transform( newX )
    #Add concepts to original features
    combined_X = hstack( ( X, csr_matrix( X1 ) ) ) 
    #Train final Logistic Regression
    self.model01 = LogisticRegression( penalty= 'l1' , C = C2, random_state=2512)
    self.model01.fit( combined_X, y )
    return

  def fitLR2( self, X, y):  
    #Fit Logistic Regression when the possible scores may vary from 0 to 2
    C1, C2, self.nComp = self.param1, self.param2, self.param3
    y1 = ( y > 0 ) * 1
    y2 = ( y > 1 ) * 1
    #Train two Logistic Regression models and repeat process described in fitLR1 with the both of them
    self.model01 = LogisticRegression( penalty= 'l1' , C = C1, random_state=2512 )
    self.model01.fit( X , y1 )
    self.model02 = LogisticRegression( penalty= 'l1' , C = C1, random_state=2512 )
    self.model02.fit( X , y2 )
    self.featSel1 = [ i for i in range(len( self.model01.coef_[0])) if self.model01.coef_[0][i] > 0 ]
    self.featSel2 = [ i for i in range(len( self.model02.coef_[0])) if self.model02.coef_[0][i] > 0 ]        

    if self.nComp == 0 or min( len( self.featSel1) , len( self.featSel2 ) ) <= self.nComp:
      return

    newX1, newX2 = X[:,self.featSel1]  , X[:,self.featSel2]
    self.concepts1 = TruncatedSVD( n_components = self.nComp, random_state = 2512 ) ## test with RBM
    self.concepts2 = TruncatedSVD( n_components = self.nComp, random_state = 2512 ) ## test with RBM
    self.concepts1.fit( newX1 )
    self.concepts2.fit( newX2 )
    X1_1, X1_2 = self.concepts1.transform( newX1 ), self.concepts2.transform( newX2 )
    X_1 , X_2  = hstack( ( X, csr_matrix( X1_1 ) ) ) , hstack( ( X, csr_matrix( X1_2 ) ) )
    self.model01 = LogisticRegression( penalty= 'l1' , C = C2 , random_state=2512 )
    self.model02 = LogisticRegression( penalty= 'l1' , C = C2 , random_state=2512 )
    self.model01.fit( X_1, y1 )
    self.model02.fit( X_2, y2 )     
    return
    
  def staged_decision_function( self, X ):
    #Return the predictions for each tree in Gradient Boosting Model
    if self.modelType == "GB2":
      return self.model01.staged_decision_function( X.toarray() )
    if self.modelType == "GB1":
      return self.model01.staged_predict_proba( X.toarray() )

  def predict_proba( self, X ):    
    #Return probabilities predicted by the model
    if self.modelType == "SV1":
      return self.model01.predict_proba ( X )

    if self.modelType == "SV2":
      mprob1 = self.model01.predict_proba( X )
      mprob2 = self.model02.predict_proba( X )
      return np.array( mprob1[:,1] ), np.array( mprob2[:,1] )

    if self.modelType == "LR2":
      if self.nComp == 0 or min( len( self.featSel1) , len( self.featSel2 ) ) <= self.nComp:
        mprob1 = self.model01.predict_proba( X )
        mprob2 = self.model02.predict_proba( X )      
      else:
        newX1 , newX2 = X[ :, self.featSel1 ], X[ :, self.featSel2 ]  
        X1_1, X1_2 = self.concepts1.transform( newX1 ), self.concepts2.transform( newX2 )
        X_1 , X_2  = hstack( ( X, csr_matrix( X1_1 ) ) ) , hstack( ( X, csr_matrix( X1_2 ) ) )
        mprob1 = self.model01.predict_proba( X_1 )
        mprob2 = self.model02.predict_proba( X_2 )
      return np.array( mprob1[:,1] ), np.array( mprob2[:,1] )

    if self.modelType == "LR1":
      if self.nComp == 0 or len( self.featSel) <= self.nComp:
        return self.model01.predict_proba ( X )
      else:
        newX = X[ :, self.featSel ]
        X1  = self.concepts.transform( newX )
        combined_X = hstack( ( X, csr_matrix( X1 ) ) ) 
        return self.model01.predict_proba( combined_X )

  def predict( self, X ):
    if self.modelType == "LR2":
      mprob1, mprob2 = self.predict_proba( X )
      yProb1 = pd.Series ( mprob1, index = range(X.shape[0]))
      yProb2 = pd.Series ( mprob2, index = range(X.shape[0]))
      yPred1 = ( yProb1 > self.tol1 ) * 1 
      yPred2 = ( yProb2 > self.tol2 ) * 2
      yPred = np.maximum( yPred1, yPred2)
      return yPred
    if self.modelType == "SV2":
      return self.model01.predict( X )

if __name__ == "__main__":
  pass