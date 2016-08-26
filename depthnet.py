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

import os
weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
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

weight_param = dict(lr_mult=1, decay_mult=0.000005)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='xavier', std=0.01),
              bias_filler=dict(type='constant', value=0)):
    conv = L.Convolution(bottom, kernel_w=ks,kernel_h=1, stride=stride,
                         num_output=nout, pad_w=pad,pad_h=0, group=group,
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

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_w=ks,kernel_h=1, stride=stride)

def caffenet(data, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data, n.label=L.HDF5Data(batch_size=50,source=data,ntop=2,shuffle=True)
    param = learned_param if learn_all else frozen_param
    
    
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 32, pad=1, param=param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3,32,stride=1, pad=1, group=1, param=param)
    
    n.pool1 = max_pool(n.relu1_2, 2, stride=2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 64, pad=1, group=1, param=param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 64, pad=1, group=1, param=param)
    
    n.pool2 = max_pool(n.relu2_2, 2, stride=2)
    
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 128, pad=1, group=1, param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 128, pad=1, group=1, param=param) 
    
    n.pool3 = max_pool(n.relu3_2, 2, stride=2)
    
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 256, pad=1, group=1, param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 256, pad=1, group=1, param=param) 
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 256, pad=1, group=1, param=param)  
    
    n.deconv4=L.Deconvolution(n.relu4_3,
                              convolution_param=dict(kernel_w=2,kernel_h=1,stride=2,num_output=128,pad_w=0,pad_h=0,group=1,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     bias_filler=dict(type='constant', value=0)),
                              param=param)                            
        
    
    n.fused_pool3=L.Eltwise(n.deconv4,n.conv3_2)
    
    n.deconv3=L.Deconvolution(n.fused_pool3,
                              convolution_param=dict(kernel_w=2,kernel_h=1,stride=2,num_output=64,pad_w=0,pad_h=0,group=1,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     bias_filler=dict(type='constant', value=0)),
                              param=param)
    
    n.fused_pool2=L.Eltwise(n.deconv3,n.conv2_2)
    
    n.deconv2=L.Deconvolution(n.fused_pool2,
                              convolution_param=dict(kernel_w=2,kernel_h=1,stride=2,num_output=32,pad_w=0,pad_h=0,group=1,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     bias_filler=dict(type='constant', value=0)),
                              param=param) 
    
    n.fused_pool2=L.Eltwise(n.deconv2,n.conv1_2)
    
    n.conv5, n.relu5 = conv_relu(n.fused_pool2, 3, 1,stride=1, pad=1, group=1, param=param)


    if not train:
        n.probs = L.Power(n.relu5)
    if n.label is not None:
        n.loss = L.EuclideanLoss(n.relu5, n.label,loss_weight=0.5)
        n.acc = L.Accuracy(n.relu5, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name
        

def style_net(train=True, learn_all=False, subset=None):

    if train:
        h5data='data/train_h5_list.txt'

    return caffenet(data=h5data, train=train,
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
