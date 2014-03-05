import glob, sys
from spellCorrector import *
from PorterStemmer import PorterStemmer
from multiprocessing import Pool

#This script preprocess the data for training the engine.

def spSymbol (text, symbol):
	#This function adds a space between and after a symbol which usually is
	#a punctuation mark.
	if( text.find( symbol ) >= 0 ):	
		text = text.replace( symbol , ' ' + symbol + ' ')
	return text

def cleanText( text ):
	
	text = text.strip().lower()
	text = text.replace( "<p>", " ")
	text = text.replace( "</p>", " ")
	text = text.replace( "</p", " ")
	text = text.replace( "<b>", " ")
	text = text.replace( "<u>", " ")
	text = text.replace( "</u>", " ")
	text = text.replace( "</b>", " ")
	text = text.replace( "<br>", " ")
	text = text.replace( "&nbsp;", " ")
	text = text.replace( "&nbsp", " ")
	text = text.replace( "&quot;", ' " ')	
	for symbol in[ ',' , '.' , '+' , '=', '-', '!','?','<','>' ]:
		text = spSymbol( text , symbol )

	while( text.find( '  ') >= 0): 
		text= text.replace('  ',' ')
	return text

def correct ( text , spellCorrectorObj):
	
	textWords = text.split()
	ftext = ""
	for word in textWords:		
		ftext = ftext + porter.stem( spellCorrectorObj.correct( word ) )  + ' '
	return ftext

def preProcess( currentFile ):
	currentName = currentFile.split("/")[-1][:-4]
	
	print "Preprocessing:\t", currentName

	Text = open( "text%s/%s.tsv" % ( stage, currentName ) , 'r' )
	outputCleanText = open( "tempText%s/%s.tsv" % ( stage, currentName ) , 'w')
	outputCleanText.write('Item_Response')
	for response in Text:
		if response[:-1] == "Item_Response" : continue
		outputCleanText.write( '\n' + cleanText( response ) ) #The text is saved to be
																															#later used to initialize the 
																															#spell corrector
	Text.close()
	outputCleanText.close()

	spellCorrectorObj = spellCorrector( currentName ) #Initialize spell corrector

	outputCleanText  = open( "tempText%s/%s.tsv"  % ( stage, currentName ) , 'r')
	outputCleanText2 = open( "tempText%s2/%s.tsv" % ( stage, currentName ) , 'w')
	outputCleanText2.write('Item_Response')
	for response in outputCleanText:
		if response[:-1] == "Item_Response" : continue
		outputCleanText2.write ( '\n' + correct( response, spellCorrectorObj ) )	#Correct spelling errors and save text
	outputCleanText.close()
	outputCleanText2.close()	

if __name__ == "__main__":
	
	global porter
	porter = PorterStemmer() 
	global stage
	stage = sys.argv[1] #It may takes "Train" or "Test"
	numCores = int( sys.argv[2] )
	
	listOfFiles = glob.glob("text%s/*.tsv" % stage)

	pool = Pool( processes = numCores )
	pool.map( preProcess, listOfFiles ) #Run in parallel
	pool.close()
	pool.join()

