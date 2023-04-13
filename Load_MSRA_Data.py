import numpy as np

  #path dhg dataset

import pandas as pd
import numpy as np
import glob

import os
import glob
import numpy
import pickle
from scipy import ndimage as ndimage
from sklearn.model_selection import train_test_split

f='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/MSRAdatast/P5/T/joint.txt'
df=np.genfromtxt(f)
#df=np.loadtxt(f)

print(df.shape)

'''
Delete 500 from all files

pattern='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/MSRAdatast/P*/*/joint.txt'
gestures_filenames = sorted(glob.glob(pattern))
#print(gestures_filenames)
for f in gestures_filenames:
    
    lines = []
    with open(f, 'r+') as fp:
         
        # read an store all lines into list
        lines = fp.readlines()
        # move file pointer to the beginning of a file
        fp.seek(0)
        # truncate the file
        fp.truncate()
        print(fp)

        # start writing lines
        # iterate line and line number
        for number, line in enumerate(lines):
            # delete line number 5 and 8
            # note: list index start from 0
            print(number, line)
            
            if number not in [0]:
                fp.write(line)

'''
def resize_gestures(input_gestures, final_length=500):
    """
    Resize the time series by interpolating them to the same length

    Input:
        - input_gestures: list of numpy.ndarray tensors.
              Each tensor represents a single gesture.
              Gestures can have variable durations.
              Each tensor has a shape: (duration, channels)
              where duration is the duration of the individual gesture
                    channels = 44 = 2 * 22 if recorded in 2D and
                    channels = 66 = 3 * 22 if recorded in 3D 
    Output:
        - output_gestures: one numpy.ndarray tensor.
              The output tensor has a shape: (records, final_length, channels)
              where records = len(input_gestures)
                   final_length is the common duration of all gestures
                   channels is the same as above 
    """
    # please use python3. if you still use python2, important note: redefine the classic division operator / by importing it from the __future__ module
    output_gestures = np.array([np.array([ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(np.size(x_i, 1))]).T for x_i in input_gestures])
    return output_gestures

pattern='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/MSRAdatast/P*/*/joint.txt'
gestures_filenames = sorted(glob.glob(pattern))
#print(gestures_filenames)
gestures = [np.genfromtxt(f) for f in gestures_filenames]
print(np.array(gestures).shape)
for i in range(1,153):
    print(np.array(gestures[i]).shape)
gestures = resize_gestures(gestures, final_length=500)
print(gestures.shape)

labels=[]
number=['1','2','3','4','5','6','7','8','9']

   

for f in gestures_filenames:
    
    k=(f.replace('\\', '/')).split('/')[-2].split('_')[0]
    
    if k in number:
         labels.append(int(k))
    elif k=='IP':
         labels.append(int(10))
    elif k=='I':
         labels.append(int(11))
    elif k=='L':
         labels.append(int(12))
    elif k=='MP':
         labels.append(int(13))
    elif k=='RP':
         labels.append(int(14))
    elif k=='T':
         labels.append(int(15))
    elif k=='TIP':
         labels.append(int(16))
    elif k=='Y':
         labels.append(int(17))
    #print(k)
    #print(labels)
    
y=np.array(labels)   
print(y.shape)
print(gestures.shape)


x_train, x_test, y_train,y_test = train_test_split(gestures, y, test_size=0.30)
#x_train, x_test, y_train_14, y_test_14= train_test_split(gestures, labels_14, test_size=0.30)

# Save the dataset
MSRA_data = {
    'x_train': x_train,
    'x_test': x_test,
    'y_train': y_train,
    'y_test': y_test,
}
import pickle
pickle_out=open("D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file/MSRA_data.pckl", "wb")
pickle.dump(MSRA_data ,pickle_out)
pickle_out.close()



f='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/MSRAdatast/P5/T/joint.txt'
k=(f.replace('\\', '/')).split('/')[-2].split('_')[0]

m=int((f.replace('\\', '/')).split('/')[-3].split(' ')[0].replace("P", ""))
print(k,m)



  
 