import numpy as np
import pandas as pd
import numpy as np
import glob

min_seq = 8

#Change the path to your downloaded SHREC2017 dataset
dataset_fold = "/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/MSRAdatast"

joint_path="/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/MSRAdatast/P*/*/joint.txt"
pattern=joint_path
gestures_filenames = sorted(glob.glob(pattern))

def split_train_test(Test_id=1):

    def label_ass(f):
        k=(f.replace('\\', '/')).split('/')[-2].split('_')[0]
        number=['1','2','3','4','5','6','7','8','9']
        if k in number:
             return int(k)
        elif k=='IP':
             return int(10)
        elif k=='I':
             return int(11)
        elif k=='L':
             return int(12)
        elif k=='MP':
             return int(13)
        elif k=='RP':
             return int(14)
        elif k=='T':
             return int(15)
        elif k=='TIP':
             return int(16)
        elif k=='Y':
             return int(17)

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
            joint_data = np.array(joint_data).reshape(21,3)#[[x1,y1,z1], [x2,y2,z2],.....]
            video.append(joint_data)
        while len(video) < min_seq:
            video.append(video[-1])
        t_id=int((f.replace('\\', '/')).split('/')[-3].split(' ')[0].replace("P", ""))
        #print('t_id=:',t_id)
        label=label_ass(f)
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

   

'''
def split_train_test(data_cfg):
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
'''
import numpy as np

def read_data_from_disk():
    def parse_data(src_file):
        video = []
        for line in src_file:
            line = line.split("\n")[0]
            data = line.split(" ")
            frame = []
            point = []
            for data_ele in data:
                point.append(float(data_ele))
                if len(point) == 3:
                    frame.append(point)
                    #print('point', np.array(point).shape)
                    point = []
            #print('frame:', np.array(frame).shape)
            video.append(frame)
        return video
    result = {}

    for g_id in range(1,15):
        print("gesture {} / {}".format(g_id,14))
        for f_id in range(1,3):
            for sub_id in range(1,21):
                for e_id in range(1,6):
                    src_path = data_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeleton_world.txt".format(g_id, f_id, sub_id, e_id)
                    src_file = open(src_path)
                    video = parse_data(src_file) #the 22 points for each frame of the video
                    #print('vedio....=',np.array(video).shape)
                    key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
                    result[key] = video
                    src_file.close()
    return result

def get_valid_frame(video_data):
    # filter frames using annotation
    info_path = data_fold + "/informations_troncage_sequences.txt"
    info_file = open(info_path)
    used_key = []
    for line in info_file:
        line = line.split("\n")[0]
        data = line.split(" ")
        g_id =  data[0]
        f_id = data[1]
        sub_id = data[2]
        e_id = data[3]
        key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
        used_key.append(key)
        start_frame = int(data[4])
        end_frame = int(data[5])
        data = video_data[key]
        video_data[key] = data[(start_frame): end_frame + 1]
        #print(key,start_frame,end_frame)
        #print(len(video_data[key]))
        #print(video_data[key][0])
    #print(len(used_key))
    #print(len(video_data))
    return video_data


def split_train_test(test_subject_id,filtered_video_data,cfg):
    #split data into train and test
    tr=0
    ts=0
    #cfg = 0 >>>>>>> 14 categories      cfg = 1 >>>>>>>>>>>> 28 cate
    train_data = []
    test_data = []
    for g_id in range(1, 15):
        for f_id in range(1, 3):
            for sub_id in range(1, 21):
                for e_id in range(1, 6):
                    key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)

                    #set table to 14 or
                    if cfg == 0:
                        label = g_id
                    elif cfg == 1:
                        if f_id == 1:
                            label = g_id
                        else:
                            label = g_id + 14

                    #split to train and test list
                    data = filtered_video_data[key]
                    sample = {"skeleton":data, "label":label}
                    if sub_id == test_subject_id:
                        test_data.append(sample)
                        ts=ts+1
                        
                    else:
                        train_data.append(sample)
                        tr=tr+1
    if len(test_data) == 0:
        raise "no such test subject"
    print('train=: ',tr,'test=:',ts)
    return train_data, test_data

def get_train_test_data(test_subject_id, cfg):
    print("reading data from desk.......")
    video_data = read_data_from_disk()
    print("filtering frames .......")
    filtered_video_data = get_valid_frame(video_data)
    train_data, test_data = split_train_test(test_subject_id,filtered_video_data,cfg) # for triaing 2660, test=140
    return train_data,test_data

#path="/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/DHGdataset/gesture_1/finger_1/subject_1/essai_1/skeleton_world.txt"
#path="./home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/DHGdataset/gesture_1/finger_1/subject_1/essai_1/skeleton_world.txt"
#src_file = open(path)
'''