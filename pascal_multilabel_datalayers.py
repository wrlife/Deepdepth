# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer
import sys
sys.path.append("./tools/Colon3D/")

import util
from scene_manager import SceneManager

from random import randint
import cv2

WORLD_SCALE=2.60721115767


class PascalMultilabelDataLayerSync(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the paramameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)
        
        self.transformer=SimpleTransformer()

        self.lrw=720/params['patch_ratio_w']
        self.lrh=240/params['patch_ratio_h']

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        if params['case']:
            top[1].reshape(self.batch_size, 1,params['im_shape'][0],  params['im_shape'][1])

        print_info("PascalMultilabelDataLayerSync", params)

    #A function randomly pick patch
    def randompatch(self,im,depth=None):
       
        
        
        params = eval(self.param_str)

        i=randint(0,240-self.lrh)
        j=randint(0,720-self.lrw)
        
        impatch=im[i:i+self.lrh,j:j+self.lrw,:]

        impatch=scipy.misc.imresize(impatch,params['im_shape'])
        
        if depth is not None:
            depthpatch=depth[i:i+self.lrh,j:j+self.lrw]
            depthpatch=cv2.resize(depthpatch,(params['im_shape'][1],params['im_shape'][0]))
            depthpatch=depthpatch[np.newaxis,:,:]
            return self.transformer.preprocess(impatch),depthpatch 
        else:

            return self.transformer.preprocess(impatch)


   
    def forward(self, bottom, top):
        """
        Load data.
        """
        params = eval(self.param_str)
        
        if params['case']:
            scene_manager = SceneManager(osp.join(params['pascal_root'], 'data/phantom/sfm_results/'))
            scene_manager.load_cameras()
            camera = scene_manager.cameras[1] # assume single camera

            # initial values on unit sphere
            x, y = camera.get_image_grid()
            r_scale = np.sqrt(x * x + y * y + 1.)

             
            for itt in range(self.batch_size):
            
                if itt % 5==0: 
                    im, multilabel = self.batch_loader.load_next_image_depth(x,y,camera)
                # Use the batch loader to load the next image.
                impatch,depthpatch=self.randompatch(im,multilabel)

                top[0].data[itt, ...] = impatch#im[:,i,:][:,np.newaxis,:]
                top[1].data[itt, ...] = depthpatch#multilabel[:,i,:][:,np.newaxis,:]

        else:
            im = self.batch_loader.load_test_image()

            #for itt in range(self.batch_size):
            count=0
            for itt in range((params['patch_ratio_h'])):
                for itj in range((params['patch_ratio_w'])):

                    impatch=scipy.misc.imresize(im[itt*self.lrh:(itt+1)*self.lrh,itj*self.lrw:(itj+1)*self.lrw,:],params['im_shape'])
                    

                    top[0].data[count, ...] = self.transformer.preprocess(impatch)
                    count=count+1



    
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.pascal_root = params['pascal_root']
        self.im_shape = params['im_shape']
        # get list of image indexes.
        list_file = params['split'] + '.txt'
        self.indexlist = [line.rstrip('\n') for line in open(
            osp.join(self.pascal_root, 'data', list_file))]
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image_depth(self,x,y,camera):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        #import pdb;pdb.set_trace();

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = index
        im = np.asarray(Image.open(
            osp.join(self.pascal_root, 'data/phantom/images', image_file_name)))
        #im = scipy.misc.imresize(im, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        #flip = np.random.choice(2)*2-1
        #im = im[:, ::flip, :]

        # Load and prepare ground truth
    
        z_gt=np.fromfile(osp.join(self.pascal_root,'data/phantom/gt_surfaces', image_file_name)+'.bin', dtype=np.float32).reshape( camera.height, camera.width)
        #z_gt=cv2.resize(z_gt,(self.im_shape[1],self.im_shape[0]))
        #z_gt=z_gt[np.newaxis,:,:]

        self._cur += 1
        return im,z_gt #self.transformer.preprocess(im), z_gt
    
    def load_test_image(self):
         # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        #import pdb;pdb.set_trace();

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = index
        im = np.asarray(Image.open(
            osp.join(self.pascal_root, 'data/phantom/test_images', image_file_name)))
        #im = scipy.misc.imresize(im, self.im_shape)  # resize

        return im #self.transformer.preprocess(im)


def load_pascal_annotation(index, pascal_root):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(21)))

    filename = osp.join(pascal_root, 'Annotations', index + '.xml')
    # print 'Loading: {}'.format(filename)

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 21), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        cls = class_to_ind[
            str(get_data_from_tag(obj, "name")).lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'index': index}


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'pascal_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])
