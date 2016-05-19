README
PremSaiKumarReddy Gangana 
(psreddy@unm.edu)
02/07/2015

Usage:
My Program is written on version Python v3.4.2 on the Integration Development Environment (IDE) IDLE v3.4.2 over operating system Windows8.1. Few dictionary methods are not compatible to previous versions 2.6 and 2.7 like my program has dict.__contains__(key) for finding out if the dictionary has a particular key. Python 2.6 and 2.7 has the same functionality achieved with dict.has_key(key). Hence to test the program, please use the pyhton version of 3.4.2 and more.

Also, My program is specific to the training and validation data provided, due to limited indexing functions for dictionaries and formatting functions on lists, I had to write code specific to this data.

To change the trainingdata or validationdata, use the lines 291 and 292. Change the filenames in the open function. However the code works only for the files with the format similar to training.txt and validation.txt

Execution:
There is no need for makefile as the code is written in a single file (Project_ID3.py). 
To execute in linux please use python3 Project_Id3.py
To execute in Windows please use IDLE environment and press F5 or execute it from command line

Note: Please use Python V3.4.2 and higher and also put the 'training.txt' and 'validation.txt' in the same folder as the code file, then the code automatically imports the file to the program without giving the path.