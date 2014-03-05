from lxml import etree
import re, glob

#This file gets the raw responses from the xml files of the test data

def cleanEnd ( text ):
  indF = 0
  for m in re.finditer(r"([A-Za-z0-9]+)", text ):
    indF = m.end()  
  text.replace( "\n", " ")
  return " ".join( text[:indF].split() )

def map_train_test( listOfFilesTrain, listOfFilesTest ):
  #This function finds a mapping between the test files
  #with the train files
  mapping = list()
  map_Test = {}
  for fileNameTest in listOfFilesTest:
    map_Test[ re.findall( r'[0-9]{5}', fileNameTest )[0] ] = fileNameTest
  for fileNameTrain in listOfFilesTrain:
    setID = re.findall( r'[0-9]{5}', fileNameTrain )[0]
    if setID in map_Test:
      mapping.append( ( fileNameTrain, map_Test[ setID ] , fileNameTrain ) )
  return mapping

if __name__ == "__main__":
  listOfFilesTrain = glob.glob("tempTextTrain/*.tsv")
  listOfFilesTest  = glob.glob("dataTest/*.xml")
  listOfFilesTrain.sort()
  listOfFilesTest.sort()

  mapping = map_train_test ( listOfFilesTrain, listOfFilesTest )

  for fileNameTrain, fileNameTest, fileNameOutput in mapping:    
    currentSet = fileNameOutput.split("/")[-1][:-4]

    outputText = open( "textTest/" + currentSet + ".tsv", 'w')
    outputText.write('Item_Response')

    all_documents = etree.parse( fileNameTest )

    for current_document in all_documents.iter():
      if current_document.tag == "Item_Response":
        outputText.write( '\n')
        outputText.write ( cleanEnd( current_document.text ) )
    outputText.close()







