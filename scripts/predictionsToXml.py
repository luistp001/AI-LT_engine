#from lxml import etree
import xml.etree.ElementTree as etree
import re, os, glob
from datetime import datetime

#This file saved the predictions in the corresponding XML files

def map_train_test( listOfFilesTrain, listOfFilesTest ):
  mapping = list()

  map_Test = {}
  for fileNameTest in listOfFilesTest:
    map_Test[ re.findall( r'[0-9]{5}', fileNameTest )[0] ] = fileNameTest
  for fileNameTrain in listOfFilesTrain:
    setID = re.findall( r'[0-9]{5}', fileNameTrain )[0]
    if setID in map_Test:
      mapping.append( ( fileNameTrain, map_Test[ setID ] , fileNameTrain ) )

  return mapping

def zero( value ):
  return "0" if value < 10 else ""

if __name__ == "__main__":
  if not os.path.isdir( "output"):
    os.mkdir( "output")
  listOfFilesTrain = glob.glob("dataTrain/*.xml")
  listOfFilesTest  = glob.glob("dataTest/*.xml")
  listOfFilesTrain.sort()
  listOfFilesTest.sort()

  mapping = map_train_test ( listOfFilesTrain, listOfFilesTest )

  for fileNameTrain, fileNameTest, fileNameOutput in mapping:

    currentSet = fileNameOutput.split("/")[-1][:-4]
    print currentSet
    
    all_documents = etree.parse( fileNameTest )
  
    scores = list( open("testPredictions/" + currentSet + ".csv") )
    
    root = all_documents.getroot()
  
    temp = root[0][0][0]
    temp.attrib = root.attrib
    all_documents._setroot( temp )
    root = all_documents.getroot()
    now = datetime.now()
    root.tag = "Job_Details"
    root.attrib['Date_Time'] =  "%d%s%d%s%d%s%d%s%d%s%d" % ( now.year, zero(now.month), now.month, zero(now.day), now.day, zero(now.hour), now.hour, zero(now.minute), now.minute, zero(now.second), now.second )
    minScore = 10

    j = 0
    for student in root:
    
      student.attrib = { 'Vendor_Student_ID' : student.attrib['Vendor_Student_ID'] }
      student[0][0].attrib['Total_CR_Item_Count'] = "1"
      student[0][0][0].tag = "Item_DataPoint_List"
      Item_ID = student[0][0][0][0].attrib['Item_ID']

      student[0][0][0][0].tag = "Item_DataPoint_Details"

      cScore = scores[j].strip() if j < len( scores ) else str(minScore)
      minScore = int(cScore) if int(cScore) < minScore else minScore

      student[0][0][0][0].attrib = { "Item_ID" : Item_ID , "Data_Point" : "", "Item_No":"1", "Final_Score":cScore}
      student[0][0][0][0][0].clear()
      student[0][0][0][0][0].tag = "Read_Details"
      student[0][0][0][0][0].attrib = { 'Read_Number':"1", "Score_Value":cScore, "Reader_ID":"1", "Date_Time": root.attrib['Date_Time'] }
      j = j + 1
  
    output = etree.tostring( root )
    output = output.replace( '\n         ', '\n')
    output = output.replace( '/></', '/>\n               </')

    outputFile = open( "output/AI-LT_" + currentSet + ".xml" , 'w')
    outputFile.write( output )
    outputFile.close()







