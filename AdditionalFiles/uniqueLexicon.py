#LT Autograder. A system that automatically grades short answer essays.
#Copyright (C) 2012 Luis Tandalla

import re

# This scripts finds the words with only lower cases in the file 'ae2.txt'
# and saves these words in the file 'ae.txt'

fileOut = open( 'ae3.txt', 'w' )
reader = open('ae.txt','r').readlines()
lexicon2 = []
for word in reader:
	wordl = word[:-1].lower()
	#print wordl
	if not wordl in lexicon2: 
		lexicon2.append ( wordl )
		fileOut.write( wordl + '\n')
