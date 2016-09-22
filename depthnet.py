# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:27:48 2016

@author: wrlife
"""
caffe_root = '/home/wrlife/bin/caffe/' 
import caffe,h5py

#caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
from pylab import *

import tempfile

from caffe import layers as L
from caffe import params as P
from caffe.coord_map import crop

import os
weights = '/home/wrlife/projects/fcn/fcn.berkeleyvision.org/voc-fcn8s/fcn8s-heavy-pascal.caffemodel'#caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
assert os.path.exists(weights)



NUM_STYLE_LABELS = 2

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

weight_param = dict(lr_mult=1, decay_mult=0.00005)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom,nout, ks=3, stride=1, pad=1, group=1,
              param=learned_param,
              weight_filler=dict(type='xavier', std=0.01),
              bias_filler=dict(type='constant', value=0)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data_layer_params,datalayer, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    if train:
        #n.data, n.label=L.HDF5Data(batch_size=30,source=datalayer,ntop=2,shuffle=True)
        n.data,n.label = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer, 
                                                  ntop = 2, param_str=str(data_layer_params))
    else:
        
   
        n.data = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer, 
                                               ntop = 1, param_str=str(data_layer_params))#n.data, n.label=L.HDF5Data(batch_size=1000,source=data,ntop=2)
        #n.label=None
    param = learned_param if learn_all else frozen_param
    
    
    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64,pad=100)  #64  pad=18
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)  #112
    #n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2) # 56
    #n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)  #28
    #n.norm3 = L.LRN(n.pool3, local_size=5, alpha=1e-4, beta=0.75)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)  #14
    #n.norm4 = L.LRN(n.pool4, local_size=5, alpha=1e-4, beta=0.75)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3) # 7
    
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    

    if train:        
        n.drop6 = fc7input = L.Dropout(n.relu6,dropout_ratio=0.5,in_place=True)
    else:
        fc7input = n.relu6
        
        
    n.fc7, n.relu7 = conv_relu(fc7input, 4096,ks=1, pad=0, param=learned_param)
    

    if train:        
        n.drop7 = Deconvinput5 = L.Dropout(n.relu7,dropout_ratio=0.5,in_place=True)
    else:
        Deconvinput5 = n.relu7
    
    
    n.score_fr = L.Convolution(Deconvinput5, num_output=21, kernel_size=1, pad=0,
        param=learned_param)
    n.upscore2 = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=21, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0)])

    n.score_pool4 = L.Convolution(n.pool4, num_output=21, kernel_size=1, pad=0,
        param=learned_param)
    n.score_pool4c = crop(n.score_pool4, n.upscore2)
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c,
            operation=P.Eltwise.SUM)
    n.upscore_pool4 = L.Deconvolution(n.fuse_pool4,
        convolution_param=dict(num_output=21, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0)])
    
#    n.score_pool3 = L.Convolution(n.pool3, num_output=21, kernel_size=1, pad=0,
#        param=learned_param)
#    n.score_pool3c = crop(n.score_pool3, n.upscore_pool4)
#    n.fuse_pool3 = L.Eltwise(n.upscore_pool4, n.score_pool3c,
#            operation=P.Eltwise.SUM)
#    n.upscore_pool3= L.Deconvolution(n.fuse_pool3,
#        convolution_param=dict(num_output=21, kernel_size=4, stride=2,
#            bias_term=False),
#        param=[dict(lr_mult=0)])
        
#    n.score_pool2 = L.Convolution(n.pool2, num_output=21, kernel_size=1, pad=0,
#        param=learned_param)
#    n.score_pool2c = crop(n.score_pool2, n.upscore_pool3)
#    n.fuse_pool2 = L.Eltwise(n.upscore_pool3, n.score_pool2c,
#            operation=P.Eltwise.SUM)
    n.upscore8 = L.Deconvolution(n.upscore_pool4,
        convolution_param=dict(num_output=21, kernel_size=16, stride=8,
            bias_term=False),
        param=[dict(lr_mult=0)])       
    
    n.score = crop(n.upscore8, n.data)
    #n.pool8 = max_pool(n.score,ks=4,stride=4)    
    n.m_score = L.Convolution(n.score, num_output=1, kernel_size=1, pad=0,
        param=learned_param)
        
    
    
    if train:
        n.loss=L.Python(n.m_score, n.label,module='myLoss',layer='Gradient_Apperance_Loss',loss_weight=0.5)        

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name
        

def style_net(train=True, learn_all=False, subset=None):
    
    
    
    pascal_root='./'
    if train:
        split='train'
        data_layer_params = dict(batch_size = 10, im_shape = [112, 112],patch_ratio_w=1,patch_ratio_h=1, split = split, pascal_root = pascal_root,case = train)
    else:
        split='test'
        data_layer_params = dict(batch_size = 1, im_shape = [120, 360],patch_ratio_w=1,patch_ratio_h=1, split = split, pascal_root = pascal_root,case = train)
#    if train:
#        h5data='nyud_train_h5_list.txt'
#    else:
#        h5data='nyud_test_h5_list.txt'

    #data_layer_params = dict(batch_size = 20, im_shape = [30, 30],patch_ratio_w=24,patch_ratio_h=8, split = split, pascal_root = pascal_root,case = train)
    #data_layer_params = dict(batch_size = 1, im_shape = [240, 720],patch_ratio_w=1,patch_ratio_h=1, split = split, pascal_root = pascal_root,case = train)
    
    return caffenet(data_layer_params,'PascalMultilabelDataLayerSync', train=train,
    #return caffenet(data_layer_params,h5data, train=train,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)


def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    #print 'top %d predicted %s labels =' % (k, name)
    #print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
    #                for i, p in enumerate(top_k))
    return top_k


def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')
    
#untrained_style_net = caffe.Net(style_net(train=False, subset='train'),
#                                weights, caffe.TEST)
#
#untrained_style_net.forward()
#style_data_batch = untrained_style_net.blobs['data'].data.copy()
#
#batch_index = 40
#image = style_data_batch[batch_index]
#plt.imshow(deprocess_net_image(image))
