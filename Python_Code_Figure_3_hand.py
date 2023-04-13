# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:06:58 2022

@author: shinlab
"""
import os
os.chdir('F:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Dataset/')

data_fold = "DHGdataset/"
import numpy as np
x=[]
y=[]
z=[]
#def read_data_from_disk():
def parse_data(src_file):
    video = []
    for line in src_file:
        line = line.split("\n")[0]
        data = line.split(" ")
        frame = []
        point = []
        for data_ele in data:
            print(data_ele)
            point.append(float(data_ele))
            if len(point) == 3:
                frame.append(point)
                #print('point', np.array(point).shape)
                x.append(point[0])
                y.append(point[1])
                z.append(point[2])
                point = []
        #print('frame:', np.array(frame).shape)
        break
        
        
        #video.append(frame)
    #return video
    return x,y,z

result = {}
#src_path = data_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeleton_world.txt".format(g_id, f_id, sub_id, e_id)
src_path = data_fold + "gesture_1/finger_1/subject_2/essai_1/skeleton_world.txt"
src_file = open(src_path)
print(src_file)
#src_file = open(src_path)
#video = parse_data(src_path) #the 22 points for each frame of the video
x,y,a = parse_data(src_file) #the 22 points for each frame of the video
print(np.array(x).shape)

from mpl_toolkits import mplot3d
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z,  cmap='red',color='red');


fig1 = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,y,z,color='red')
plt.show()




    
'''
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
                    break
            break
        break
    
    return result
'''


from mpl_toolkits import mplot3d
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')


# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
ax = plt.axes(projection='3d')

zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Red');