from lxml import etree
import re, glob, sys

#This file gets the raw responses and the scores from the xml files

def cleanEnd ( text ):
	indF = 0
	for m in re.finditer(r"([A-Za-z0-9]+)", text ):
		indF = m.end()	
	text.replace( "\n", " ")
	return " ".join( text[:indF].split() )

if __name__ == "__main__":
	trainType = "Train"

	listOfFiles = glob.glob("data%s/*.xml" % trainType )

	for fileName in listOfFiles:
		currentSet = fileName.split("/")[-1][:-4]

		outputText  = open( "text%s/%s.tsv" % ( trainType, currentSet ) , 'w')
		outputScore = open( "score/%s.csv" % ( currentSet ) , 'w')
		outputText.write('Item_Response')
		outputScore.write('Final_Score,score1,score2')

		all_documents = etree.parse( fileName )
	
		prev_Student_Test_ID = 0
		for current_document in all_documents.iter():

			if current_document.tag == "Student_Test_Details":
				Student_Test_ID = current_document.attrib[ "Student_Test_ID"]

			if current_document.tag == "Item_Response":
				outputText.write( '\n')
				outputText.write ( cleanEnd( current_document.text ) )

			elif current_document.tag == "Item_DataPoint_Score_Details":
				if Student_Test_ID == prev_Student_Test_ID:
					continue

				outputScore.write( '\n'+ current_document.attrib [ 'Final_Score' ] )
				numScores = 0	
				for partial_doc in current_document.iter():				
					if partial_doc.tag == 'Score':
						numScores += 1
						if numScores > 2: break
						if 'Score_Value' in partial_doc.attrib:
							outputScore.write( ','+ partial_doc.attrib['Score_Value'] )
						else:
							outputScore.write( ',-1' )
				prev_Student_Test_ID = Student_Test_ID








