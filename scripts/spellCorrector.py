#LT Autograder. A system that automatically grades short answer essays.
#Copyright (C) 2012 Luis Tandalla

import re, collections, sys
#This file uses code from http://norvig.com/spell-correct.html

def words(text): return re.findall('[a-z]+', text.lower()) 
  #It returns all the words from a text

def train(features):
  #It returns two dictionaries with the counts of words 
  #and bigrams contained in the file features
    model = collections.defaultdict(lambda: 1)
    model2 = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    for i in range(len(features) - 1):
      f = features[i] +  " " + features[i+1]
      model2[f] += 1
    return model, model2


class spellCorrector:

  def __init__(self, fileName="", SET = '0'):
    # It creates lexicon (list of right words)
    # It creates two dictionaries with the counts of words and bigrams
    self.fileName = fileName
    self.NWORDS, self.NWORDS2 = train(words(file('AdditionalFiles/big.txt').read())) #It calculates counts of words and bigrams
                                                                     #from file 'big.txt'
                                                                     
    self.lexicon = open('AdditionalFiles/ae3.txt','r').readlines()
    self.lexicon = { word[:-1] for word in self.lexicon }

    self.eliminateMispelledToTrain()

    self.small_lexicon = set() # Lexicon of words already corrected

    self.alphabet = "'abcdefghijklmnopqrstuvwxyz"
    self.cache = {}

  def eliminateMispelledToTrain( self ):
    #It create two dictionaries with the counts of words and bigrams from the file 
    #located in tempTextTrain.
    model = collections.defaultdict(lambda: 1)
    model2 = collections.defaultdict(lambda: 1)

    text = str( file( "tempTextTrain/" + self.fileName + ".tsv").read() )
    allWords = text.split()
    prevWord = ""
    for word in allWords:
      
      if word in self.lexicon:
        model[word] += 1        
        if prevWord:
          model2[ prevWord + " " + word ] += 1
        prevWord = word
      else:
        prevWord = ""

    self.localModel = model    
    self.localModel2 = model2
    self.localModel_list = set( [ word for word in self.localModel if self.localModel[ word ] > 2 ] )


  def in_lexicon( self, word ):
    #It checks if a word is a right word (a not mispelled word).
    return word in self.lexicon

  def edits1(self,word):
   # It returns a set with words that have an edit distance of 1 with the 'word'
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in self.alphabet]

   a = set(deletes + transposes + replaces + inserts)
   return a  

  def known_edits2(self,word,b,c):
    # It returns two sets of right words. The words of one set have edit distance 
    # of 2 with the 'word' and the words of the other set have edit distance of 1. 
    # The original word is also included in both sets.

    a = set(e2 for e1 in b for e2 in self.edits1(e1) if e2 in self.small_lexicon or e2 in self.NWORDS )

    return (a.union(c)).union( set([word]) )

  def known(self,words): return set(w for w in words if w in self.small_lexicon or self.in_lexicon( w ))
    # It returns a set of right words that have edit distance of 1 with the 'word'

  def known_edits1(self, word):
    b = self.edits1(word)
    c = self.known(b)
    return b,c    

  def correct_2(self,word, wordp = "", wordn = "", type = 1):

    b , c      = self.known_edits1( word ) 
    candidates = c.union( set( [ word ] )  )

    #Returns a possible word if it is in the local model
    possible_answer = max(candidates, key=self.localModel.get) 
    if possible_answer in self.localModel_list:     
      self.cache[ word ] = possible_answer
      return possible_answer

    candidates2 = self.known_edits2( word, b, c)
    possible_answer2 = max(candidates2, key=self.localModel.get) 
    if possible_answer2 in self.localModel_list:    
      self.cache[ word ] = possible_answer2
      return possible_answer2
    
    #Returns a possible word if it is in the NWORDS dictionary
    possible_answer = max(candidates, key=self.NWORDS.get) 
    if self.in_lexicon( possible_answer ):          
      self.cache[ word ] = possible_answer
      return possible_answer         
      
    if len(possible_answer) >=4 :
      splits = [(word[:i+2], word[i+2:]) for i in range(len(word) - 3)] # It produces splits of the word           
      for word1, word2 in splits:
       if self.in_lexicon(word1) and self.in_lexicon(word2):  # If both words of a split are right, 
        self.cache[ word ] = word1 + ' ' + word2
        return word1 + ' ' + word2                            # it returns the split
                 
      pos_answ = ['','','']
      for word1, word2 in splits: 
        # Finds the splits that have one word right, but keeps the split with the longest
        # right word. Then, it corrects the other word and returns the split.
       if len( pos_answ[0]) < len(word1) and self.in_lexicon(word1) : pos_answ = [word1, word1, word2 ]
       if len( pos_answ[0]) < len(word2) and self.in_lexicon(word2) : pos_answ = [word2, word1, word2 ]

      if not pos_answ[0] =='' : 
        pword1, pword2 = self.correct( pos_answ[1] ) , self.correct( pos_answ[2] )
        if self.in_lexicon( pword1 ) or self.in_lexicon( pword2 ):
          possible_split = pword1 + ' ' + pword2
          self.cache[ word ] = possible_split
          return possible_split
    self.cache[ word ] = possible_answer 
    return possible_answer # It the program goes to this point, it means 
                                                       # that the original word could not be corrected,
                                                       # so possible word is the same as the original word.

  def contain_digits( self, word ):
    return len ( re.findall( r'[0-9]', word) ) > 0

  def ispunc( self, word ):
    return word in [ ',' , '.' , '+' , '=', '-', '!','?','<','>' ]

  def correct(self,word, wordp = "", wordn = ""):
    
    if word in self.small_lexicon:    return word
    if len( word ) > 15: return word
             
    if word in self.cache:     return self.cache[word]
    if self.in_lexicon( word ): 
      self.small_lexicon.add ( word )
      return word
    if self.contain_digits( word ):   return word
    if self.ispunc ( word ) :         return word
            
    temp = self.correct_2( word, wordp, wordn)
    return temp

if __name__ == "__main__":
    pass
    

