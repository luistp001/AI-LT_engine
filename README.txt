Training the hyper-parameters of the engine:

  The training files with the short answer responses should
  be placed in the folder textTrain as tsv files. They
  are expected to have a header.

  The scores of those files should be placed in the score folder. 
  It is expected that they have three columns for three scores with
  the first column being the main score. A header is also expected.

  -Run the file trainComplete.sh with the command
   ./trainComplete.sh

  The parameters numCores and level can be modified in the script.
  More description can be found in the file itself.

  The file will train the engine with different combination of parameters
  and save those that produce the highest kappa. It will train the engine for
  each of the files located in the textTrain folder.
  

Training the engine:

  -Train the engine with ./train.sh

  It requires the same files required for ./trainComplete.sh. It also requires
  the parameters saved by ./trainComplete.sh

  The ./train.sh will train the engine and save all the required files
  to make the predictions for the scores.


Testing the engine:

  The test files with the short answer responses should
  be placed in the folder textTest as tsv files. They
  are expected to have a header.

  -Run the file ./test.sh
  It expects the files saved by ./train.sh and save the
  predicted scores in testPredictions.



