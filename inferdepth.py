# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 23:54:14 2016

@author: wrlife
"""
#import caffe
import matplotlib.pyplot as plt
import cv2
from depthnet import *
from defsolver import *
WORLD_SCALE = 2.60721115767

style_weights="weights.pretrained.caffemodel"

test_net = caffe.Net(style_net(train=True,learn_all=False),style_weights, caffe.TEST)
test_net.forward()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X,Y=np.mgrid[:227,:227];
surf = ax.plot_surface(X,Y,test_net.blobs['label'].data[20][0,:,:])

#surf = ax.plot_surface(X,Y,test_net.blobs['deconv1'].data[20][0,:,:],color='r')
#surf = ax.plot_surface(X,Y,ttt,color='r')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X,Y=np.mgrid[:227,:227];
surf = ax.plot_surface(X,Y,test_net.blobs['deconv1'].data[20][0,:,:],color='r')
plt.show()
#for i in range (0, test_net.blobs['label'].data.shape[0]-1):
#    
#    plt.plot(test_net.blobs['label'].data[i][0,0,:])
#
#    plt.plot(test_net.blobs['conv5'].data[i][0,0,:])
#
#    savefig('data/sfs/est/'+'{:4d}'.format(i)+'.png')
#    
#    close()


est_z=np.zeros((100,200))
for i in range (0,100):
    est_z[i,:]=test_net.blobs['conv5'].data[i][0,0,:]



import sys
sys.path.append("./tools/Colon3D/")

import util
from scene_manager import SceneManager

colmap_folder='/home/wrlife/projects/mydltest/data/phantom/sfm_results/'
scene_manager = SceneManager(colmap_folder)
scene_manager.load_cameras()
camera = scene_manager.cameras[1] # assume single camera
# initial values on unit sphere
x, y = camera.get_image_grid()
r_scale = np.sqrt(x * x + y * y + 1.)


est_z=cv2.resize(est_z,(camera.width, camera.height))
est_z=est_z.reshape(camera.height, camera.width, 1)

util.save_sfs_ply('%s_%s.ply' % ('test', 'depth'),np.dstack((x, y, np.ones_like(x)))/WORLD_SCALE * est_z.reshape(camera.height, camera.width, 1))

#model_folder='/home/wrlife/projects/mydltest/data/phantom/gt_surfaces/'
#filename='frame0850.jpg.bin'
#
#test_d=np.fromfile(model_folder + filename, dtype=np.float32).reshape(camera.height, camera.width, 1)
#
#util.save_sfs_ply('%s_%s.ply' % ('test', 'depth'),np.dstack((x, y, np.ones_like(x)))/WORLD_SCALE *test_d)
#

