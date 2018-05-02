'''
This is a standalone implementation of Kabsch Algorithm
w.r.t.

'''

import numpy as np
import random
import math
from math import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def createData(N):
    
    data = np.array([[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]])
    
#     random.seed(10)
#     data = np.zeros((N,3), dtype=np.float32)
#     
#     cnt = 0
#     
#     for i in range(N):
#         for j in range(3):
#             data[i,j] = cnt
#             cnt += 1
    
    return data

def runKabsch(P, Q):
    print('running Kabsch...')
    # P - rotated dataset, Q - reference dataset
    
    # 1. find centroid of two datasets and perform translation (omit here)
    # 2. calculate covariance matrix 
    A = np.dot(np.transpose(P), Q)
    U,S,Vt = np.linalg.svd(A)
    print(S)
    R = np.dot(U, Vt)
    
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(U, Vt)
        print('\tremoved reflection')
    
    print('estimated rotation matrix: ')
    print(R)
    
    return R

def createRotMat3D(angleX=0, angleY=0, angleZ=0):
    
    rotX = np.array([[1,0,0], [0, cos(angleX), -sin(angleX)], [0, sin(angleX), cos(angleX)]])
    rotY = np.array([[cos(angleY), 0, -sin(angleY)], [0, 1, 0], [sin(angleY), 0, cos(angleY)]])
    rotZ = np.array([[cos(angleZ), -sin(angleZ), 0], [sin(angleZ), cos(angleZ), 0], [0,0,1]])
    
    return np.dot(rotZ, np.dot(rotY, rotX))


def errRMS(A, B):
    '''
    root-mean-square error between datasets
    '''
    
    assert A.shape == B.shape
    
    return np.sum(np.sqrt(np.sum(np.power((A-B),2), axis=1)), axis=0) / A.shape[0]


if __name__ == '__main__':
    
    '''
    TODO:
    1. axis-angle representation
    2. inverse of matrix might be unstable
    3. translate by centroid
    '''
    
    rotMat = createRotMat3D(0, math.pi/2) # create a known rotation matrix
    print('original rotation matrix: ')
    print(rotMat)
    
    samples = createData(4) # create N sample scene points
    
    print('original samples: ')
    print(samples)
    
    samples_rot = np.transpose(np.dot(rotMat, np.transpose(samples))) # apply the known rotation and fake known data points
    print('rotated samples: ')
    print(samples_rot)
    
    estRotMat = runKabsch(np.copy(samples_rot), np.copy(samples)) # obtain estimated rotation by Kabsch algo
    
    estRotMatInv = np.linalg.inv(estRotMat) # inverse of estimated rotation matrix
    
    samples_est = np.transpose(np.dot(estRotMatInv, np.transpose(samples_rot))) # apply inverse rotation to known data points
    print('estimated samples: ')
    print(samples_est)
    
    print('root mean squares: ')
    print('\t samples vs rotated_samples: {}'.format(errRMS(samples, samples_rot)))
    print('\t samples vs estimated_samples: {}'.format(errRMS(samples, samples_est)))
    
    
    cstr = ['r', 'g', 'b', 'cyan']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[:,0], samples[:,1], samples[:,2], c=cstr, marker='x', s=100)
    ax.scatter(samples_rot[:,0], samples_rot[:,1], samples_rot[:,2], c=cstr, marker='o', s=20)
    ax.scatter(samples_est[:,0], samples_est[:,1], samples_est[:,2], c=cstr, marker='^', s=50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])
    
#     ax = fig.add_subplot(122, projection='3d')
#     ax.scatter(samples_rot[:,0], samples_rot[:,1], samples_rot[:,2], c=cstr, marker='x')
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     ax.set_xlim([-2,2])
#     ax.set_ylim([-2,2])
#     ax.set_zlim([-2,2])
    
    #plt.show()
    
    pass