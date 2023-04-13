""" This is for Shric data"""


# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:25:49 2022

@author: shinlab
"""
'''
# ---------------------------------------------------------
# Step 1. Download hand gesture datasets
# ---------------------------------------------------------
download_shrec_17 = False
download_dhg = True
download_online_dhg = False

# --------------------------
# SHREC2017 dataset
#     http://www-rech.telecom-lille.fr/shrec2017-hand/
# --------------------------
if download_shrec_17:
  !mkdir dataset_shrec2017
  !wget http://www-rech.telecom-lille.fr/shrec2017-hand/HandGestureDataset_SHREC2017.tar.gz -O SHREC2017.tar.gz
  !tar -xzf SHREC2017.tar.gz -C dataset_shrec2017
# --------------------------
# DHG14/28 dataset
#     http://www-rech.telecom-lille.fr/DHGdataset/
# --------------------------
# Note: you should register on http://www-rech.telecom-lille.fr/DHGdataset/ before downloading the dataset
if download_dhg:
  !mkdir dataset_dhg1428
  !wget http://www-rech.telecom-lille.fr/DHGdataset/DHG2016.zip
  !unzip DHG2016.zip -d dataset_dhg1428
# --------------------------
# Online DHG dataset
#     http://www-rech.telecom-lille.fr/shrec2017-hand/
      http://www-rech.telecom-lille.fr/shrec2017-hand/
# --------------------------
if download_online_dhg:
  !mkdir dataset_onlinedhg
  !wget http://www-rech.telecom-lille.fr/shrec2017-hand/OnlineDHG.zip
  !unzip OnlineDHG.zip -d dataset_onlinedhg
  '''
  #path dhg dataset
path_dhg='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/DHGdataset'
path_shrec17='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/Shrdataset'
#path_online_dhg=
path_pickle='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file'
# ---------------------------------------------------------
# Step 2. Utils
# ---------------------------------------------------------
import os
import glob
import numpy
import pickle
from scipy import ndimage as ndimage
from sklearn.model_selection import train_test_split


def resize_gestures(input_gestures, final_length=100):
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
    output_gestures = numpy.array([numpy.array([ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(numpy.size(x_i, 1))]).T for x_i in input_gestures])
    return output_gestures

y_14=[]
y_28=[]
def load_gestures(dataset='shrec', root=path_shrec17, version_x='3D', version_y='both', resize_gesture_to_length=100):
#def load_gestures(dataset='dhg', root=path_dhg, version_x='3D', version_y='both', resize_gesture_to_length=100):
    """
    Get the 3D or 2D pose gestures sequences, and their associated labels.

    Ouput:
        - a tuple of (gestures, labels) or (gestures, labels_14, labels_28)
              where gestures is either a numpy.ndarray tensor or
                                       a list of numpy.ndarray tensors,
                                       depending on if the gestures have been resized or not.
              Each tensor represents a single gesture.
              Gestures can have variable durations.
              Each tensor has a shape: (duration, channels) where channels is either 44 (= 2 * 22) or 66 (=3 * 22)
    """

    # SHREC 2017 (on Google Colab):
    # root = '/content/dataset_shrec2017/HandGestureDataset_SHREC2017'
    # DHG 14/28 (on Google Colab):
    # root = '/content/dataset_dhg1428'
    if dataset == 'dhg':
       root=path_dhg
    if dataset == 'shrec':
      #assert 'dataset_shrec' in root
      root=path_shrec17
    
    if version_x == '3D':
        if dataset == 'dhg':
            pattern = root + '/gesture_*/finger_*/subject_*/essai_*/skeleton_world.txt'
            #pattern=os.path.join(root, '/gesture_*/finger_*/subject_*/essai_*/skeleton_world.txt')
        elif dataset == 'shrec':
            pattern = root + '/gesture_*/finger_*/subject_*/essai_*/skeletons_world.txt'
            #pattern=os.path.join(root,'/gesture_*/finger_*/subject_*/essai_*/skeletons_world.txt')
            #print(pattern)
    else:
        if dataset == 'dhg':
            pattern = root + '/gesture_*/finger_*/subject_*/essai_*/skeleton_image.txt'
        elif dataset == 'shrec':
            pattern = root + '/gesture_*/finger_*/subject_*/essai_*/skeletons_image.txt'

    gestures_filenames = sorted(glob.glob(pattern))
    gestures = [numpy.genfromtxt(f) for f in gestures_filenames]
    if resize_gesture_to_length is not None:
        gestures = resize_gestures(gestures, final_length=resize_gesture_to_length)
    #f=gestures_filenames[1]
    #print(f)
    #f=f.replace('\\', '/')
    #print(f)
    #print('musa')
    #print(f.split('/')[-5].split('_')[1])
    #print(f.split('/')[-4].split('_')[1])



    labels_14 = [int((filename.replace('\\', '/')).split('/')[-5].split('_')[1]) for filename in gestures_filenames]
    #y_14=labels_14

    labels_28 = [int((filename.replace('\\', '/')).split('/')[-4].split('_')[1]) for filename in gestures_filenames]
    
    labels_28 = [labels_14[idx_gesture] if n_fingers_used == 1 else 14 + labels_14[idx_gesture] for idx_gesture, n_fingers_used in enumerate(labels_28)]
    #y_28=labels_28
    
    labels_14[:] = [number - 1 for number in labels_14]
    labels_28[:] = [number - 1 for number in labels_28]

    if version_y == '14' or version_y == 14:
        return gestures, labels_14
    elif version_y == '28' or version_y == 28:
        return gestures, labels_28
    elif version_y == 'both':
        return gestures, labels_14, labels_28


def write_data(data, filepath):
    """Save the dataset to a file. Note: data is a dict with keys 'x_train', ..."""
    with open(filepath, 'wb') as output_file:
        pickle.dump(data, output_file)


def load_data(filepath=path_pickle+'/shrec_data.pckl'):
    """
    Returns hand gesture sequences (X) and their associated labels (Y).
    Each sequence has two different labels.
    The first label  Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with your index pointed, or not pointed).
    """
    file = open(filepath, 'rb')
    data = pickle.load(file, encoding='latin1')  # <<---- change to 'latin1' to 'utf8' if the data does not load
    file.close()
    return data['x_train'], data['x_test'], data['y_train_14'], data['y_train_28'], data['y_test_14'], data['y_test_28']


# ---------------------------------------------------------
# Step 3. Save the dataset(s) you need
# ---------------------------------------------------------
# Example: 3D version of the SHREC17 and DHG gesture datasets, with gestures resized to 100 timesteps

gestures, labels_14, labels_28 = load_gestures(dataset='shrec',
                                               root=path_shrec17,
                                               version_x='3D',
                                               version_y='both',
                                               resize_gesture_to_length=100)
"""
#Split the dataset into train and test sets if you want:
gestures, labels_14, labels_28 = load_gestures(dataset='dhg',
                                               root=path_dhg,
                                               version_x='3D',
                                               version_y='both',
                                               resize_gesture_to_length=100)
"""
x_train, x_test, y_train_14, y_test_14, y_train_28, y_test_28 = train_test_split(gestures, labels_14, labels_28, test_size=0.30)
#x_train, x_test, y_train_14, y_test_14= train_test_split(gestures, labels_14, test_size=0.30)

# Save the dataset
data = {
    'x_train': x_train,
    'x_test': x_test,
    'y_train_14': y_train_14,
    'y_train_28': y_train_28,
    'y_test_14': y_test_14,
    'y_test_28': y_test_28
}
write_data(data, filepath='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file/shrec_data.pckl')
#write_data(data, filepath='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file/dhg_data.pckl')
import numpy as np

print("Xtrain=",np.array(x_train).shape)
print("Xtest=",np.array(x_test).shape)
print("Ytrain14=",np.array(y_train_14).shape)
print("Ytrain28=",np.array(y_train_28).shape)
print("Ytest14=",np.array(y_test_14).shape)
print("Ytest28=",np.array(y_test_28).shape)
# ---------------------------------------------------------
# Step 4. Optional: copy to google drive, if you're in a Google Colab
# ---------------------------------------------------------
try:

  # Connect Google Colab instance to Google Drive
  #from google.colab import drive
  #drive.mount('/gdrive')
  shrec_data=data
  import pickle
  pickle_out=open("D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file/shrec_data.pickle", "wb")
  pickle.dump(shrec_data ,pickle_out)
  pickle_out.close()
    #
    #

  # Save your dataset on Google Drive
  #!cp dhg_data.pckl /gdrive/My\ Drive/dhg_data.pckl

  # Load your dataset from Google Drive
  # !cp /gdrive/My\ Drive/dhg_data.pckl dhg_data.pckl

except:
  print("You're not in a Google Colab!")
  
  
  
  
  # ---------------------------------------------------------
# Step 5. Use the dataset(s)
# ---------------------------------------------------------
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = load_data('D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file/shrec_data.pckl')
  
  
#link: https://colab.research.google.com/drive/1ggYG1XRpJ50gVgJqT_uoI257bspNogHj#scrollTo=zDKQ_UL6yF75


print(np.array(y_14).shape)
a=[bin for bin  in range(30)]

import numpy as np

hist, bins = np.histogram(y_test_14, a)
 
# printing histogram
print('Printing histogram')
print (hist)
print (bins)
print()



p=path_shrec17='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/Shrdataset/gesture_11/finger_2/subject_27/essai_1/skeletons_world.txt'

labels_14 = int((p.replace('\\', '/')).split('/')[-5].split('_')[1]) 
    #y_14=labels_14

n_fingers_used = int((p.replace('\\', '/')).split('/')[-4].split('_')[1])
if n_fingers_used==1:
   labels_28= labels_14
else:
    labels_28= 14+labels_14
    
subject=int((p.replace('\\', '/')).split('/')[-3].split('_')[1])

print('gesture_14, gesture_28, subject',labels_14, labels_28, subject)