import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.cross_validation import KFold, StratifiedKFold, Bootstrap
import skllMetrics, re, random
from sklearn.base import clone
  
#This function contains the functions that load the data and extract features 
#from the text

def get_data( currentSet, ngramMax, kInter , shuffle = 0, returnSubset = False):
  #This function reads the files with the short answer responses and the scores
  text, originalText = get_text ( "tempTextTrain2/" + currentSet + ".tsv" , ngramMax, kInter )
  score = pd.read_csv("score/" + currentSet + ".csv")  

  text, originalText, y, y1, y2, subset = get_valid_data( text, originalText, score , shuffle) 
  if returnSubset: 
    return text, originalText, y, y1, y2, subset
  else:
    return text, originalText, y, y1, y2

def get_data_test( currentSet, ngramMax, kInter ):
  #This function reads the files with the short answer responses of the test data
  text, originalText = get_text ( "tempTextTest2/" + currentSet + ".tsv" , ngramMax, kInter )
  return text

def rangeY ( currentSet, ngramMax =0, kInter=0 ):
  #This function returns the maximum possible score for the current set.
  text, originalText, y, y1, y2 = get_data ( currentSet, ngramMax, kInter )
  return np.max( y )

def get_XTrain_XTest ( text, y, cv = 5):
  #This function returns a matrix split in Train and Test data.

  #This line initializes the class that will change the list of texts to matrices. It
  #applies logarithm to the counts and considers only the features that have a 
  #minimum count of 2
  modelT = tfidf( analyzer = 'word', ngram_range = (1,1) , token_pattern = r'[^ ]+', min_df = 2 ,
                  norm = None, use_idf = False, smooth_idf = False, sublinear_tf = True )


  indices_Train, indices_Test = list(), list()
  XTrain, XTest = list(), list()
  cvI = StratifiedKFold(y, cv, indices= True)

  for train, test in cvI:
    indices_Train.append( train )
    indices_Test.append ( test  )
    
    textTrain = [ text[i] for i in train ]
    textTest  = [ text[i] for i in test  ]
    
    modelC = clone( modelT )
    modelC.fit( textTrain )

    XTrain.append( modelC.transform( textTrain ) )
    XTest.append ( modelC.transform( textTest  ) )

  return XTrain, XTest, indices_Train, indices_Test


def get_text( fileName , ngramMax, kInter ):
  #This function opens the file with the texts and return a list with
  #a modified version of each short answer responses
  inputFile = open( fileName, 'r' )
  listText = list()
  for line in inputFile:
    listText.append( line.strip() )
  inputFile.close()
  originalText = listText[1:]
  text = [ val.replace('er then', 'er than') for val in originalText]
  text = [ val.replace("\\", '/') for val in text]  
  text = [ changeText(val, ngramMax, kInter) for val in text ]
  return text, originalText

def changeText( text, ngramMax , kInter  ):  
  #This function changes the text by adding unigrams, bigrams,
  #up to ngramMax-grams. It also adds the bigrams with (kInter - 1)
  #intermediate words
  words = re.findall( r'[^ ]+' , text)
  textWords = list() 
  for ngramI in range( 1, ngramMax + 1 ):
    for i in range( len( words ) - ngramI + 1 ):
      textWords.append( "_".join ( words[i:( i+ ngramI  )] ) )

  if kInter > 1:
    for i in range( len( words ) ):
      for j in range(2, kInter+1):
        if (i + j) >= len( words ): continue
        textWords.append( words[i] + "_" + words[i+j] )
  return " ".join( textWords )

def get_valid_data( text, originalText, score , shuffle ) :
  nx = len(text)
  y , y1, y2 = score['Final_Score'], score['score1'] , score['score2']
  random.seed(2512)
  #If shuffles the data "shuffle" times
  for s in range(shuffle):
    textS = list()
    yS, yS1, yS2 = y.copy(), y1.copy(), y2.copy()
    sampleInd = random.sample( range(len(text)), len(text), )
    for j in range(len(text)):
      textS.append( text[ sampleInd[j] ] )
      yS.ix[j]  = y.ix[ sampleInd[j] ]
      yS1.ix[j] = y1.ix[ sampleInd[j] ]
      yS2.ix[j] = y2.ix[ sampleInd[j] ]
      pass
    text = textS
    y, y1, y2 = yS.copy(), yS1.copy(), yS2.copy()

  subset = [ i for i in range(nx) if str( y.ix[i] ).isdigit() and str( y1.ix[i] ).isdigit() and str( y2.ix[i] ).isdigit() ]
  y, y1, y2 = y[subset], y1[subset], y2[subset]
  text  = [ text[i] for i in subset ]
  originalText = [ originalText[i] for i in subset ]
  for i in y.index:
    y.ix[i]  = int( y.ix[i])
    y1.ix[i] = int(y1.ix[i])
    y2.ix[i] = int(y2.ix[i])
  return text, originalText, y, y1, y2, subset



def getParams ( currentSet, scale = None ):
  #This function is used in the training scripts to load the saved parameters after training the engine
  paramsLines = open( "modelResults/selected/%s.txt" % currentSet)
  params = { line.strip().split(":")[0] : line.strip().split(":")[1].split() for line in paramsLines if line.strip() }
  for modelType in params:
    params[modelType][0] = int( params[modelType][0] ) 
    params[modelType][1] = int( params[modelType][1] ) 
    params[modelType][2] = int( params[modelType][2] )
    if modelType == "GB":
      params[modelType][3] = int( params[modelType][3] )
      params[modelType][4] = float( params[modelType][4] )
    else:
      params[modelType][3] = float( params[modelType][3] )
      params[modelType][4] = float( params[modelType][4] ) if int( params[modelType][2] ) > 0 else 0.0
    if scale <= 2:
      params[modelType] = params[modelType][:5]
    else:
      params[modelType] = params[modelType][:(5+scale)] 
  return params