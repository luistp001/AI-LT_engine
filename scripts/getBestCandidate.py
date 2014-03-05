import glob
#This script gets prints the combination of parameters that produce the highest kappa
#for the GB and SV models.

def processFile( currentSet , typeN , outputFile ):
  try:
    inputFile = open( "modelResults/%s_topk/%s.txt" % ( typeN, currentSet) , 'r')  
    outputFile.write( typeN + ":")
    bestKappa, bestLine = 0, None
  
    for line in inputFile:
      lineS = line.strip().split()
      if len( lineS ) == 11:
        currentKappa = float( lineS[7] )
      if len( lineS ) == 10:
        currentKappa = float( lineS[6] )
      if currentKappa > bestKappa:
        bestKappa, bestLine = currentKappa, line
  
    outputFile.write( bestLine )
    inputFile.close()
  except IOError:
    return  

def main():
  
  listOfAllFiles = glob.glob("textTrain/*.tsv")
  listOfAllFiles.sort()

  ltypeN = ["GB", "SV"]  
  for fileName in listOfAllFiles:
    currentSet = fileName.split("/")[-1][:-4]
    outputFile = open( "modelResults/selected/%s.txt" % currentSet, 'a')

    for typeN in ltypeN:
      processFile( currentSet , typeN , outputFile ) 

    outputFile.close()
    
if __name__ == "__main__":
  main()