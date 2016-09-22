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
from itertools import izip

from scene_manager import SceneManager
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
KD_TREE_KNN = 10

# bilinearly interpolates depth values for given (non-integer) 2D positions
# on a surface
def get_estimated_r(S, points2D_image):
  r_est = np.empty(points2D_image.shape[0])

  for k, (u, v) in enumerate(points2D_image):
    # bilinear interpolation of distances for the fixed 2D points on the current
    # estimated surface
    j0, i0 = int(u), int(v) # upper left pixel for the (u,v) coordinate
    udiff, vdiff = u - j0, v - i0 # distance of sub-pixel coord. to upper left
    p = (udiff * vdiff * S[i0,j0,:] + # this is just the bilinear-weighted sum
      udiff * (1 - vdiff) * S[i0+1,j0,:] +
      (1 - udiff) * vdiff * S[i0,j0,:] +
      (1 - udiff) * (1 - vdiff) * S[i0+1,j0+1,:])

    r_est[k] = np.linalg.norm(p)

  return r_est

def nearest_neighbor_warp(weights, idxs, points2D_image, r_fixed, S):
  # calculate corrective ratios as a weighted sum, where the corrective ratios
  # relate the fixed to estimated depths
  r_ratios = r_fixed / get_estimated_r(S, points2D_image)

  w = np.sum(weights * r_ratios[idxs], axis=-1) / np.sum(weights, axis=-1)
  w = gaussian_filter(w, 7)

  # calculate corrective ratios as a weighted sum
  S *= w[:,:,np.newaxis]

  return S, r_ratios

style_weights="weights.pretrained.caffemodel"

test_net = caffe.Net(style_net(train=False,learn_all=False),style_weights, caffe.TEST)
test_net.forward()

#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#X,Y=np.mgrid[:120,:120];
#surf = ax.plot_surface(X,Y,test_net.blobs['label'].data[20][0,:,:])
#
##surf = ax.plot_surface(X,Y,test_net.blobs['deconv1'].data[20][0,:,:],color='r')
##surf = ax.plot_surface(X,Y,ttt,color='r')
#plt.show()

#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#X,Y=np.mgrid[:120,:120];
#surf = ax.plot_surface(X,Y,test_net.blobs['m_score'].data[2][0,:,:],color='r')
#plt.show()
#for i in range (0, test_net.blobs['label'].data.shape[0]-1):
#    
#    plt.plot(test_net.blobs['label'].data[i][0,0,:])
#
#    plt.plot(test_net.blobs['conv5'].data[i][0,0,:])
#
#    savefig('data/sfs/est/'+'{:4d}'.format(i)+'.png')
#    
#    close()


lrh=240/2
lrw=720/6


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


x=cv2.resize(x,(360,120))
y=cv2.resize(y,(360,120))
util.save_sfs_ply('%s_%s.ply' % ('test', 'depth9'),np.dstack((x, y, np.ones_like(x)))/WORLD_SCALE * test_net.blobs['m_score'].data[0].transpose(1,2,0))

x=cv2.resize(x,(100,100))
y=cv2.resize(y,(100,100))

x=cv2.pyrDown(x)
y=cv2.pyrDown(y)

#est_z=np.zeros((21,70))
#points2D=np.zeros((21*70,2))
#r_fix=np.zeros((21*70))
#count=0
#for i in range (0,21):
#    for j in range (0,70):
#      util.save_sfs_ply('%s_%s.ply' % ('test', 'depth222'),np.dstack((x, y, np.ones_like(x)))/WORLD_SCALE * ttt.transpose(1,2,0))
#      est_z[i,j] = test_net.blobs['fc9'].data[count][0,0,0]
#      count=count+1
#      points2D[i*70+j]=np.array([j*10+15,i*10+15])
#      r_fix[i*70+j]=np.linalg.norm(np.array([x[i*10+15,j*10+15]*est_z[ i,j],y[i*10+15,j*10+15]*est_z[i,j],est_z[i,j]]))
#    test_net.forward()
#    count=0

lrh=240/8
lrw=720/24

est_z=np.zeros((21,70))
points2D=np.zeros((21*70,2))
r_fix=np.zeros((21*70))

image_file_name='frame0850.jpg'
z_gt=np.fromfile(osp.join('./data/phantom/gt_surfaces', image_file_name)+'.bin', dtype=np.float32).reshape( camera.height, camera.width)

#est_z=np.zeros((1,240,720))
#count=0
#for i in range (0,8):
#    for j in range (0,24):
#      
#      est_z[:,i*lrh:(i+1)*lrh,j*lrw:(j+1)*lrw] = test_net.blobs['m_score'].data[count]
#      count=count+1
#    count=0
#    if i!=7:
#        test_net.forward()
#
#count=0
#for i in range (0,21):
#    for j in range (0,70):
#      
#      est_z[i,j] = test_net.blobs['m_score'].data[count][0,lrh/2,lrw/2]      
#      count=count+1
#      points2D[i*70+j]=np.array([j*10+lrw/2,i*10+lrh/2])
#      r_fix[i*70+j]=np.linalg.norm(np.array([x[i*10+lrh/2,j*10+lrw/2]*est_z[ i,j],y[i*10+lrh/2,j*10+lrw/2]*est_z[i,j],est_z[i,j]]))
#    count=0
#    if i!=20:
#        test_net.forward()
        
        
#count=0
#for i in range (0,90):
#    for j in range (0,30):
#        est_z[i,j]
#        
#      
#      est_z[i,j] = test_net.blobs['m_score'].data[count][0,lrh/2,lrw/2]      
#      count=count+1
#      points2D[i*70+j]=np.array([j*10+lrw/2,i*10+lrh/2])
#      r_fix[i*70+j]=np.linalg.norm(np.array([x[i*10+lrh/2,j*10+lrw/2]*est_z[ i,j],y[i*10+lrh/2,j*10+lrw/2]*est_z[i,j],est_z[i,j]]))
#    count=0
#    if i!=20:
#        test_net.forward()

#util.save_sfs_ply('%s_%s.ply' % ('test', 'depth1'),np.dstack((x, y, np.ones_like(x)))/WORLD_SCALE * est_z.transpose(1,2,0))
#S=est_z.transpose(1,2,0)

#depthz=cv2.resize(est_z,(camera.width, camera.height))
#est_z=est_z.reshape(camera.height, camera.width, 1)


 #print 'Computing nearest neighbors'

points2D_image = points2D.copy()
points2D = np.hstack((points2D, np.ones((points2D.shape[0], 1))))
points2D = points2D.dot(np.linalg.inv(camera.get_camera_matrix()).T)[:,:2]

kdtree = KDTree(points2D)
weights, nn_idxs = kdtree.query(np.c_[x.ravel(),y.ravel()], KD_TREE_KNN)
weights = weights.reshape(camera.height, camera.width, KD_TREE_KNN)
nn_idxs = nn_idxs.reshape(camera.height, camera.width, KD_TREE_KNN)
 
   # create initial surface on unit sphere
S0 = np.dstack((x, y, np.ones_like(x)))
S0 /= np.linalg.norm(S0, axis=-1)[:,:,np.newaxis]
S = S0.copy()
z = S0[:,:,2]
  
S, r_ratios = nearest_neighbor_warp(weights, nn_idxs,
        	    points2D_image, r_fix, util.generate_surface(camera, z))
  
util.save_sfs_ply('%s_%s.ply' % ('warp', 'depth1'), S)


#util.save_sfs_ply('%s_%s.ply' % ('test', 'depth'),np.dstack((x, y, np.ones_like(x)))/WORLD_SCALE * depthz.reshape(camera.height, camera.width, 1))

#model_folder='/home/wrlife/projects/mydltest/data/phantom/gt_surfaces/'
#filename='frame0850.jpg.bin'
#
#test_d=np.fromfile(model_folder + filename, dtype=np.float32).reshape(camera.height, camera.width, 1)
#
#util.save_sfs_ply('%s_%s.ply' % ('test', 'depth'),np.dstack((x, y, np.ones_like(x)))/WORLD_SCALE *test_d)
#

