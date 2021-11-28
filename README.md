# CancerCellTracker
The algorithm proposed in manuscript "CancerCellTracker: A Brightfield Time-lapseMicroscopy Framework for Cancer DrugSensitivity Estimation" can detect, track, and classify  cancer cells in time-lapse microscopy images. 

## Environment
This program was developped under Ubuntu with Python 3.8.8. The following python packages are required.
- Numpy
- scipy
- Opencv
- astropy
- statistics

## Cell Classification
The following files are the codes required to classify cancer cells into live and dead.

### **cell_detect.py**
This file implements detection of cells in the images. It contains several steps such as converting color images to gray scale images and gray scale images to binary images, finding contours in binary images, determining if a contour is actually a cell, and calculating the shape of the cells.
### **cell_classify.py**
This code classifies cells in the images. It contains several steps such as tracking cells through contiuous images, determining which cells are live and which are dead.
### **cell_process.py**
This is the entry of the program. This code takes time-lapse microscopy images as input data and gives the user the classfication of cells as output. It calls cell_detect.py and cell_classify.py for the computation. 

### How to Run
Users need to run the code in Ubuntu Environment. After download the repository, run the following command under the folder ./codes:

$./cell_process.py ./configure.txt ../dataset/sample_1/ ../output/


