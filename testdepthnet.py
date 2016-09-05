# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:49:08 2016

@author: wrlife
"""

#from TwoDdepthnet import *
from depthnet import *
from defsolver import *

niter = 1000  # number of iterations to train

# Reset style_solver as before.
base_lr = 0.00000000001
test_net=style_net(train=True,learn_all=True)

#test_net = caffe.Net(style_net(train=True), caffe.TRAIN)

style_solver_filename = solver(test_net,base_lr=base_lr)
style_solver = caffe.get_solver(style_solver_filename)
style_solver.net.copy_from(weights)

print 'Running solvers for %d iterations...' % niter

solvers = [('pretrained', style_solver)]

loss, acc, weights = run_solvers(niter, solvers)
print 'Done.'

style_weights = weights['pretrained']


import os

os.system("mv "+style_weights+" ./")

del style_solver, solvers
