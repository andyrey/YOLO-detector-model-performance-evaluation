import os
import pandas as pd
from os import walk, getcwd
from datetime import datetime
end_diff = '_gt'
file_end = 'txt'

CLASSES = ['car', 'plate_number', 'headlight', 'broken_headlight', 'wheel',
           'flat_tire', 'door', 'windshield', 'doorknob', 'rearlight']
CLASSES_NUMBER = len(CLASSES)         
IOU_THRESHOLD = 0.3
model_name = "YOLOv4custom"

def dir_walk(dir_path):
    '''
    reads all txt files in the dir and forms a list of their names
    '''
    file_name_list_ = []
    file_name_list_txt = []
    for (dirpath, dirnames, filenames) in walk(dir_path):
        file_name_list_.extend(filenames)
        break  # ??
    for each_file in file_name_list_:
        try:
            bare_name, ext = (each_file).split('.')
        except ValueError:
            continue
        if ext == file_end:
            file_name_list_txt.append(each_file)
    return file_name_list_txt


def joint_gt_detec_pairs(filenamelist):
    '''
    takes list of file names and makes list  of file pairs
    [(file_detect1.txt, file_gt1.txt),(file_detect2.txt, file_gt2.txt),...]
    '''
    file_names_detected = []  # [filename in filenamelist if filename =="ff"]
    file_names_gt = []
    for filename in filenamelist:
        if end_diff in filename:
            file_names_gt.append(filename)
        else:
            file_names_detected.append(filename)
    if len(file_names_detected) > len(file_names_gt):
        print("Detections files are more than ground truth!")
    elif len(file_names_detected) < len(file_names_gt):
        print("Detections files are less than ground truth!")
    else:
        print("Detections files amount == that of ground truth!")
    file_names_detected.sort()
    file_names_gt.sort()
    # = [(gt, d) for gt in file_names_gt if d in file_names_detected and d + end_diff == gt]
    joint_list = []
    ending_length = len(end_diff) + len(file_end) + 1
    for gt in file_names_gt:
        detect = gt[: -ending_length]+"." + file_end
        if detect in file_names_detected:
            joint_list.append((gt, detect))
        else:
            print(f"Detection file {detect} is absent!")
    return joint_list


def sort_lines_in_file(file):
    '''
    input csv file, output sorted by class (0-9) list of lists of BB coordinates
    convert strings into list of lists
    '''
    line_array = []  # [[]]
    with open(file, 'r') as txtfile:
        # lines = file.read().split('\n')  # for ubuntu, use "\r\n" instead of "\n"??
        lines = txtfile.readlines()
        for each_line in lines:
            if each_line == '':
                break
            l = each_line.split(' ')
            l[1:] = [float(i) for i in l[1:]]
            line_array.append(l)
    line_array.sort(key=lambda x: x[0])  # sort by first element= class
    return line_array


def calc_iou(r1, r2):
    '''
    calculates intersection over union of two rectangles
    '''
    # convert from (xc,yc,w,h) to (x1,y1,x2,y2)
    boxA = (r1[0] - r1[2]/2, r1[1] - r1[3]/2,
            r1[0] + r1[2]/2, r1[1] + r1[3]/2)
    boxB = (r2[0] - r2[2]/2, r2[1] - r2[3]/2,
            r2[0] + r2[2]/2, r2[1] + r2[3]/2)
# calculation for 0.0 --1.0 relative coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea <= 0 or boxBArea <= 0:  # can't be <0, just precausions
        return 0
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calc_performance(fp, fn, tp, classes_involed):
    '''
    takes False Positive, False Negative and True Positive and return recall and precision for 1 frame
    (1 pair of gt and detection)
    '''
    recall = [None]*CLASSES_NUMBER
    precision = [None]*CLASSES_NUMBER
    for i in range(CLASSES_NUMBER):
    #    if str(i) not in classes_involed:
    #        continue
  
        if tp[i] == 0:
            recall[i] = 0
            precision[i] = 0
        else:
            recall[i] = tp[i]/(tp[i] + fn[i])
            precision[i] = tp[i]/(tp[i] + fp[i])
    return (recall, precision)


def process_1_files_pair(files_pair):  # gt + detection files 1 pair tuple
    '''
    main function: takes gt + detection files 1 pair tuple, for each line in  gt finds the best correspondence in lines from detection,
    and returns False Positive, False Negative and True Positive performance
    Nested cycles: for Class, for each line of the Class, for each line of the detection
    '''
    lines_gt_full = sort_lines_in_file(files_pair[0]) #all lines
    lines_det_full = sort_lines_in_file(files_pair[1])
    # pick up all classes in current gt and detection file pair:
    kk_gt = [row[0] for row in lines_gt_full]
    kk_det = [row[0] for row in lines_det_full]
    kk_gt.extend(kk_det)# учитываем классы не только в эталоне, но и появившиеся (ошибочно) в детекции
    
    # unique set of all classes in this ground true file+ those which are only in detections
    current_classes = sorted(set(kk_gt))
    #file may contain object class not interesting for us
    current_classes = [k for k in current_classes if int(k) < CLASSES_NUMBER]

    tp = [0]*CLASSES_NUMBER  # true positive
    fp = [0]*CLASSES_NUMBER
    fn = [0]*CLASSES_NUMBER
    iou_threshold = IOU_THRESHOLD
    # Zero loop**************************************************************************************************************
    for classnom in range(CLASSES_NUMBER):
        dic = dict()  # {(line1#,line2#):iou}
        #lets leave only lines with current classnom:
        lines_gt = [line for line in lines_gt_full if line[0] == str(classnom)]
        lines_det= [line for line in lines_det_full if line[0] == str(classnom)]
        line1_count = 0  # номер строки данного класса, если их много
        # -1st loop****************************************************************************
        for line1 in lines_gt:  # for each line in gt file - 1st loop
            if str(classnom) == line1[0]:
                line2_count = 0
                # -2nd loop*********************************************************
                for line2 in lines_det:  # compare with each line with same class in detections
                    if str(classnom) == line2[0]:
                        iou = calc_iou(line1[1:], line2[1:])
                        # add element to dict
                        dic[(line1_count, line2_count)] = iou
                        line2_count += 1
                # end of 2nd loop****************************************************

                line1_count += 1
        # End of 1st loop***********************************************************************
        # набрали пары всевозможных комбинаций строк одного класса classnom,ищем комбинацию с макс  iou-наилучшее соответсвие,
        # удаляем все остальные пары, где хотя бы одна из этих строк присутствует.
        #all possible gt and detection combination found. Let's remove redundand
        counted_gt = 0
        counted_det =0
        while bool(dic):  # nonempty
            # coртировка по значению iou в пределах одного класса
            max_key = max(dic, key=dic.get)# показывает ключ к макс value, напр (0,1)->iou
            if dic[max_key] > iou_threshold:
                tp[classnom] += 1
                #del lines_gt[max_key[0]]
                counted_det +=1
                counted_gt  +=1
                # removes found correspondencies both from lists and dict
                #del lines_det[max_key[1]]
# удаляет из словаря излишние элементы, где iou не максимальное, а ключ содержит один из элементов line1_number или line2_number,
# которые уже попали в пару с максимальным iou и таким образом проучаствовали:
#leave only those elements with max iou and get rid of those have been considered:
            dic = {key: val for (key, val) in dic.items() if key[0] != max_key[0] and key[1] != max_key[1]}
            #End of while loop
            # leftovers represents fp and fn
        fp[classnom] += sum(1 for line_ in lines_det if line_[0] == str(classnom))
        fn[classnom] += sum(1 for line_ in lines_gt  if line_[0] == str(classnom))  # len(lines_gt)
        fp[classnom] -= counted_det
        fn[classnom] -= counted_gt
        # end of zero loop************************************************************************************************************
    return (fp, fn, tp, current_classes)

cwd = getcwd()
fnl = dir_walk(cwd)
file_pair_list = joint_gt_detec_pairs(fnl)
#i = 0
class_count = [0]*CLASSES_NUMBER
Recall_mean = [0]*CLASSES_NUMBER
Precision_mean = [0]*CLASSES_NUMBER
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)	
df_common_header_framely_output = pd.DataFrame(columns=[str(CLASSES_NUMBER)+" classes",model_name,dt_string,"","",""])
df_common_header_framely_output.to_csv('out_framely.csv', mode = 'a')
#with open('../result.txt', 'w') as fr:
for file_pair in file_pair_list:
    #fr.write("File {}\nclass   FP      FN       TP    Recall    Precision\n".format(file_pair[1]))
    cols = ['   Class','FP','FN','TP','Recall','Precision']
    df_empty_string = pd.DataFrame(columns=["","","","","",""])
    df_file_name_string = pd.DataFrame(columns=["File ", file_pair[1],"","","",""])
    df_perform_params_framely = pd.DataFrame(columns= cols)
    FP, FN, TP, classes_here = process_1_files_pair(file_pair)
    Recall, Precision = calc_performance(FP, FN, TP, classes_here)
      
    for j in range(CLASSES_NUMBER):
        if str(j) in classes_here:
            Recall_mean[j] = (Recall_mean[j]*class_count[j] + Recall[j])/(class_count[j]+1)
            Precision_mean[j] = (Precision_mean[j]*class_count[j] + Precision[j])/(class_count[j]+1)
            class_count[j] += 1
      
    for cl in classes_here:
        #fr.write(f"  {cl}     {FP[int(cl)]}    {FN[int(cl)]}      {TP[int(cl)]}     {Recall[int(cl)]:.3f}      {Precision[int(cl)]:.3f}\n")
        df_perform_params_framely.loc[int(cl)] = pd.Series( {'   Class':CLASSES[int(cl)], 'FP':FP[int(cl)], 'FN':FN[int(cl)], 'TP':TP[int(cl)],'Recall':Recall[int(cl)],'Precision':Precision[int(cl)]})
    df_empty_string.to_csv('out_framely.csv', mode = 'a')
    df_file_name_string.to_csv('out_framely.csv', mode = 'a')
    df_perform_params_framely.to_csv('out_framely.csv', mode = 'a')
#End of output file:
sRecall_mean = ['None' if class_count[i] <=1 else str(round(r,3)) for i, r in enumerate(Recall_mean)  ]
sPrecision_mean = ['None' if class_count[i] <=1 else str(round(r,3)) for i, r in enumerate(Precision_mean)  ]

#now form resulting csv file

#I stayed this if there is need to print as txt:
#fr.write(f"\n              0       1      2     3     4      5      6     7     8     9\n")
#fr.write(f"Recall_mean   {sRecall_mean[0]} {sRecall_mean[1]} {sRecall_mean[2]} {sRecall_mean[3]} {sRecall_mean[4]} \
#{sRecall_mean[5]} {sRecall_mean[6]} {sRecall_mean[7]} {sRecall_mean[8]} {sRecall_mean[9]} \n")
df_headline_result_file = pd.DataFrame(columns=["Model", str(CLASSES_NUMBER)+" classes",model_name,dt_string,"","",""])
df_performance_params_result_file =  pd.DataFrame(columns=[CLASSES[i] for i in range(CLASSES_NUMBER) ])#[CLASSES[0],CLASSES[1],CLASSES[2],CLASSES[3],CLASSES[4],CLASSES[5],CLASSES[6],CLASSES[7],CLASSES[8],CLASSES[9]])
df_performance_params_result_file.loc["Recall mean"]= pd.Series({ CLASSES[i]:sRecall_mean[i] for i in range(CLASSES_NUMBER) })
df_performance_params_result_file.loc["Precision mean"]= pd.Series({ CLASSES[i]:sPrecision_mean[i] for i in range(CLASSES_NUMBER) })
df_headline_result_file.to_csv('out_general.csv')
df_performance_params_result_file.to_csv('out_general.csv', mode='a')











