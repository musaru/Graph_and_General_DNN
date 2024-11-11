import numpy as np
import pandas as pd
import numpy as np
import glob
min_seq=8
dataset_fold = "/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/Shrdataset"
joint_path="/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/Shrdataset/gesture_*/finger_*/subject_*/essai_*/skeletons_world.txt"
pattern=joint_path
gestures_filenames = sorted(glob.glob(pattern))

def split_train_test(data_cfg,Test_id=1):
    def label_ass(path, data_cfg):
        if data_cfg==0:
           labels_14 = int((path.replace('\\', '/')).split('/')[-5].split('_')[1]) 
           return labels_14
        else:
            labels_14 = int((path.replace('\\', '/')).split('/')[-5].split('_')[1]) 
            n_fingers_used = int((path.replace('\\', '/')).split('/')[-4].split('_')[1])
            if n_fingers_used==1:
               labels_28= labels_14
            else:
                labels_28= 14+labels_14
            return labels_28      
    ts=0
    tr=0
    label_list = []
    all_data = []
    train_data = []
    test_data=[]
    for f in gestures_filenames:  
        video=[]
        joint_file=open(f)    
        for joint_line in joint_file:
            joint_data = joint_line.split()
            joint_data = [float(ele) for ele in joint_data]#convert to float
            joint_data = np.array(joint_data).reshape(22,3)#[[x1,y1,z1], [x2,y2,z2],.....]
            video.append(joint_data)
        while len(video) < min_seq:
            video.append(video[-1])
        t_id=int((f.replace('\\', '/')).split('/')[-3].split('_')[1])
        #print('t_id=:',t_id)
        label=label_ass(f,data_cfg)
        sample = {"skeleton":video, "label":label}
        if t_id==Test_id:
           test_data.append(sample)
           ts=ts+1
        else:
            train_data.append(sample)
            tr=tr+1  
            
    if len(test_data) == 0:
        raise "no such test subject"
    print('train=: ',tr,'test=:',ts)  
    return train_data, test_data    

    
    
    


#train_path = dataset_fold + "/train_gestures.txt"
#train_file = open(train_path)
'''
def split_train_test(data_cfg,Test_id=1):
    def parse_file(data_file,data_cfg):
        #parse train / test file


        label_list = []
        all_data = []
        for line in data_file:
            data_ele = {}
            data = line.split() #【id_gesture， id_finger， id_subject， id_essai， 14_labels， 28_labels size_sequence】
            #video label
            if data_cfg == 0:
                label = int(data[4])
            elif data_cfg == 1:
                label = int(data[5])
            label_list.append(label) #add label to label list
            data_ele["label"] = label
            #video
            video = []
            joint_path = dataset_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(data[0],data[1],data[2],data[3])
            joint_file = open(joint_path)
            for joint_line in joint_file:
                joint_data = joint_line.split()
                joint_data = [float(ele) for ele in joint_data]#convert to float
                joint_data = np.array(joint_data).reshape(22,3)#[[x1,y1,z1], [x2,y2,z2],.....]
                video.append(joint_data)
            while len(video) < min_seq:
                video.append(video[-1])
            data_ele["skeleton"] = video
            data_ele["name"] = line
            all_data.append(data_ele)
            joint_file.close()
        return all_data, label_list



    print("loading training data........")
    train_path = dataset_fold + "/train_gestures.txt"
    train_file = open(train_path)
    train_data, train_label = parse_file(train_file,data_cfg)
    assert len(train_data) == len(train_label)

    print("training data num {}".format(len(train_data)))

    print("loading testing data........")
    test_path = dataset_fold + "/test_gestures.txt"
    test_file = open(test_path)
    test_data, test_label = parse_file(test_file, data_cfg)
    assert len(test_data) == len(test_label)

    print("testing data num {}".format(len(test_data)))

    return train_data, test_data
'''