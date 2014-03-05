import glob, sys
#This script organizes the parameters in the output of the trainTune script.
#It prints the combination of parameters that produce the highest kappa for each set
#New output is saved in modelResults/ _topk/ directory

def processFile( fileName , modelType ):
  inputFile = open( fileName , 'r' )
  outputFile, scale = None, None
  bestParams, bestKappa = None, 0

  for line in inputFile:
    lineS = line.strip().split()
    if len( lineS ) > 2 and lineS[2] == "8":
      outputFile.write( bestParams )
      bestParams, bestKappa = None, 0
    
    if not lineS:
      continue
    if lineS[0] == "set":
      if bestParams:
        outputFile.write( bestParams )
        bestParams, bestKappa = None, 0
      if outputFile:
        outputFile.close()
      outputFile = file( "modelResults/%s_topk/%s.txt" % ( modelType, lineS[1] ) , 'w')
      scale = int( lineS[4] )
      continue
    if len( lineS ) == 2: #Reset parameters
      if bestParams:
        outputFile.write( bestParams )
      bestParams, bestKappa = None, 0
      continue
    if not lineS[0].isdigit():
      continue

    currentKappa = lineS[ scale + 5 ] 
    if float( currentKappa ) > bestKappa: #Update parameters
      bestKappa, bestParams = float( currentKappa ), line
      
  if bestParams:
    outputFile.write( bestParams )
    bestParams, bestKappa = None, 0
  if outputFile:
    outputFile.close()
      
def main():
  modelType = sys.argv[1]

  fileList = glob.glob("modelOutput/%s/*" % modelType )
  fileList.sort()

  for fileName in fileList:
    print fileName
    processFile( fileName , modelType) 
    
if __name__ == "__main__":
  main()