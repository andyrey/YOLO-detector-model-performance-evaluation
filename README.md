# YOLO-detector-model-performance-evaluation
This Python program evaluates performance of YOLO (v3,v4) detecting model from comparison test and ground truth files, yielding TP, FP, FN, Recall and Precision output.
How to use:

1. Elaborate your files with YOLO  detections (like 00000_0000000715.txt as example) *.txt and put them in same folder with corresponding ground truth files with matched names *_gt.txt so they are paired (see test_and_gt_folder.jpg). The order of detections and annotations lines in the pairs may be different- algorithm finds the best matches. The number of lines in detections and annotation files may be different- this will be considered in FP and FN. 
  The first column is number of class and next floats are bounding box coordinates in YOLO format.
  You should set desirable IOU_THRESHOLD (0.3 is enough in my case), so you get AP30 or that of you need for your benchmark.

2. Put there Python script "eval.py" also. Edit lines 8 -12 according your needs.

3. Launch python3 eval.py. 

4. You will get False Positive, False Negative,True Positive and Recall and Precision for each file for each class in file out_framely.csv.
Also you will get total (mean) Recall and Precision for each class in file out_general.csv.
