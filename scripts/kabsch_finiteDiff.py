import torch
import math
from math import cos, sin


def createData(N):

    data = torch.FloatTensor(
        [[1., 0., 0.],
         [0., 0., 0.],
         [0., 1., 0.],
         [1., 1., 0.]
         ])

    return data


def createRotMat3D(angleX=0, angleY=0, angleZ=0):

    rotX = torch.FloatTensor(
        [[1, 0, 0],
         [0, cos(angleX), -sin(angleX)],
         [0, sin(angleX), cos(angleX)]])

    rotY = torch.FloatTensor(
        [[cos(angleY), 0, -sin(angleY)],
         [0, 1, 0],
         [sin(angleY), 0, cos(angleY)]])

    rotZ = torch.FloatTensor(
        [[cos(angleZ), -sin(angleZ), 0],
         [sin(angleZ), cos(angleZ), 0],
         [0, 0, 1]])

    return torch.mm(rotZ, torch.mm(rotY, rotX))


def runKabsch(P, X, autograd_=False):
    print('running Kabsch...')
    # P - known data, X - scene point variables

    # 1. find centroid of two datasets and perform translation (omit here)
    # 2. calculate covariance matrix

    if autograd_:
        X.requires_grad = True
    else:
        X.requires_grad = False

    A = torch.mm(torch.t(P), X)

    U, S, V = torch.svd(A)

    Vt = torch.t(V)

    R = torch.mm(U, Vt)

    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = torch.mm(U, Vt)
        print('\tremoved reflection')

    print('estimated rotation matrix: ')
    print(R)

    if autograd_:
        R.backward(torch.FloatTensor(
            [[1, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]), retain_graph=True)
        return [X.grad.data] # delta(r(1,1)) / delta(X)

    return [R]


def computeDiff(A, B):

    if A.dim() == 1:
        print('Dim=1')
        return torch.sum(torch.pow((A-B), 2), 0)
    elif A.dim() == 2:
        print('Dim=2')
        return torch.sum(torch.sum(torch.pow((A-B), 2), 1), 0)


if __name__ == '__main__':

    rotMat = createRotMat3D(0, math.pi/2)  # create a known rotation matrix

    samples = createData(4)  # create 4 sample scene points

    samples_rot = torch.t(torch.mm(rotMat, torch.t(samples)))

    eps = 0.1
    N = 4

    res_fd = []

    res_comp = []

    # test: calculate w.r.t to a particular element X[i,j]
    for i in [3]:
        for j in [2]:

            samples[i, j] += eps
            forward = runKabsch(samples_rot, samples.clone())

            samples[i, j] -= 2*eps
            backward = runKabsch(samples_rot, samples.clone())

            for k in range(len(forward)):
                res_fd.append(forward[k].sub(backward[k]) / (2*eps))

            samples[i, j] += eps
            res_ag = runKabsch(samples_rot, samples, True)

            #
            print('Compute gradient difference:')
            for k in range(len(res_fd)):
                print(res_fd[k])
                print(res_ag[k])
                #res_comp.append(computeDiff(res_fd[k], res_ag[k]))

            print(res_comp)
