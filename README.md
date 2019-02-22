1. You may have to change the paths in test.txt and dji.txt according to your work directory.

- The testing images for this model is in the ./sample folder.
  If you only need to run the evaluation, just change the paths in the test.txt

- The training images for this object detection task is available for download here:
  https://pitt.app.box.com/s/756141768nn92cj0dkfbg6dan17c4h4q
  
  After downloading the data, the paths in dji.txt needs to be adjusted accordingly.

- I did not provide a script to adjust the paths.. if you need one, please let me know.
 

2. To run the test only:

python train.py -model=iSmart2DNN -weight=iSmart2DNN.weights -test=test.txt -train=dji.txt -eval


3. To train the network:

Just remove the option "-eval"


4. If you need the HLS code as well, please let me know.


*** My Contact ***
hc.onioncc@gmail.com


