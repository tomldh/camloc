'''
Created on 2 May 2018

@author: fighterlin
'''

import torch
import math
from math import cos, sin

from torch.autograd import Variable

def createData(N):
    
    data = Variable(torch.FloatTensor([[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]]), requires_grad=True)
    
    return data

def createRotMat3D(angleX=0, angleY=0, angleZ=0):

    rotX = Variable(torch.FloatTensor([[1,0,0], [0, cos(angleX), -sin(angleX)], [0, sin(angleX), cos(angleX)]]), requires_grad=False)
    
    rotY = Variable(torch.FloatTensor([[cos(angleY), 0, -sin(angleY)], [0, 1, 0], [sin(angleY), 0, cos(angleY)]]), requires_grad=False)
    
    rotZ = Variable(torch.FloatTensor([[cos(angleZ), -sin(angleZ), 0], [sin(angleZ), cos(angleZ), 0], [0,0,1]]), requires_grad=False)
    
    return torch.mm(rotZ, torch.mm(rotY, rotX))

def runKabsch(P, X):
    print('running Kabsch...')
    # P - known data, X - scene point variables
    
    # 1. find centroid of two datasets and perform translation (omit here)
    # 2. calculate covariance matrix 
    
    A = torch.mm(torch.t(P), X)
    U,S,Vt = torch.svd(A)
    print(S)
    R = torch.mm(U, Vt)
    
    if torch.det(R) < 0:
        Vt[2,:] *= -1
        R = torch.mm(U, Vt)
        print('\tremoved reflection')
    
    print('estimated rotation matrix: ')
    print(R)
    print(R.requires_grad)
    
    return R

def errRMS(A, B):
    '''
    root-mean-square error between two datasets
    '''
    #assert A.shape == B.shape
    
    return torch.sum(torch.sqrt(torch.sum(torch.pow((A-B),2), axis=1)), axis=0) / A.shape[0]

if __name__ == '__main__':
    
    rotMat = createRotMat3D(0, math.pi/2) # create a known rotation matrix
    print('original rotation matrix: ')
    print(rotMat)
    print(rotMat.requires_grad)
    
    samples = createData(4) # create N sample scene points
    
    print('original samples: ')
    print(samples)
    print(samples.requires_grad)
    
    samples_rot = torch.t(torch.mm(rotMat, torch.t(samples))) # apply the known rotation and fake known data points
    print('rotated samples: ')
    print(samples_rot)
    print(samples_rot.requires_grad)
    
    
    estRotMat = runKabsch(samples_rot.clone(), samples.clone()) # obtain estimated rotation by Kabsch algo
    
    estRotMatInv = torch.inverse(estRotMat) # inverse of estimated rotation matrix
    
    samples_est = np.t(torch.mm(estRotMatInv, np.t(samples_rot))) # apply inverse rotation to known data points
    print('estimated samples: ')
    print(samples_est)
    
    print('root mean squares: ')
    print('\t samples vs rotated_samples: {}'.format(errRMS(samples, samples_rot)))
    print('\t samples vs estimated_samples: {}'.format(errRMS(samples, samples_est)))
    
    pass