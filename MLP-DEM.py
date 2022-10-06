import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from sklearn.utils import shuffle
import datetime
import h5py
from mpl_toolkits.mplot3d import Axes3D
import torch
print(torch.__version__)
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib as mpl
import numpy.random as npr
import scipy.integrate as sp
from pyevtk.hl import gridToVTK
import pandas as pd 
import numpy.linalg as la
from torch.multiprocessing import Process, Pool
from NumIntg import *
# import rff
import pyvista as pv
torch.manual_seed(2022)
mpl.rcParams['figure.dpi'] = 350

torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    device_string = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device_string = 'cpu'
    print("CUDA not available, running on CPU")



def setup_domain():
    x_dom = 0, Length, Nx
    y_dom = 0, Width,  Ny
    z_dom = 0, Depth,  Nz
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    lin_z = np.linspace(z_dom[0], z_dom[1], z_dom[2])
    domEn = np.zeros((Nx * Ny * Nz, 3))
    c = 0
    for z in np.nditer(lin_z):
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c
            te = tb + y_dom[2]
            c += 1
            domEn[tb:te, 0] = x
            domEn[tb:te, 1] = lin_y
            domEn[tb:te, 2] = z
    print('Uniform Nodes', domEn.shape)
    np.meshgrid(lin_x, lin_y, lin_z)

    dom = domEn
    
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx = np.where(dom[:, 0] == 0)
    bcl_u_pts = dom[bcl_u_pts_idx, :][0]
    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where(dom[:, 0] == Length)
    bcr_t_pts = dom[bcr_t_pts_idx, :][0]
    bcr_t_pts_idx_uniform = np.where(domEn[:, 0] == Length)
    bcr_t_pts_uniform = domEn[bcr_t_pts_idx_uniform, :][0]
    top_idx = np.where((dom[:, 1]==Width) & (dom[:, 0]>0) & (dom[:, 0]<Length))
    top_pts = dom[top_idx, :][0]
    bottom_idx = np.where((dom[:, 1] == 0) & (dom[:, 0]>0) & (dom[:, 0]<Length))
    bottom_pts = dom[bottom_idx, :][0]
    front_idx = np.where((dom[:, 2] == Depth) & (dom[:, 0]>0) & (dom[:, 0]<Length))
    front_pts = dom[front_idx, :][0]
    back_idx = np.where((dom[:, 2] == 0) & (dom[:, 0]>0) & (dom[:, 0]<Length))
    back_pts = dom[back_idx, :][0]
    allnodes = np.arange(len(dom))
    inn_nodes_indx = np.setdiff1d(allnodes,bcr_t_pts_idx)
    inn_nodes_indx = np.setdiff1d(inn_nodes_indx,bcl_u_pts_idx)
    inn_nodes_indx = np.setdiff1d(inn_nodes_indx,top_idx)
    inn_nodes_indx = np.setdiff1d(inn_nodes_indx,bottom_idx)
    inn_nodes_indx = np.setdiff1d(inn_nodes_indx,front_idx)
    inn_nodes_indx = np.setdiff1d(inn_nodes_indx,back_idx)
    inn_nodes = dom[inn_nodes_indx,:]


    domain           = {}
    domain['Domain'] = torch.from_numpy(dom).float()
    domain['Energy'] = torch.from_numpy(domEn).float()
    domain['Xint']   = torch.from_numpy(inn_nodes_indx).long()
    domain['X1']     = torch.from_numpy(bcr_t_pts_idx[0]).long()
    domain['X1_Uni'] = torch.from_numpy(bcr_t_pts_idx_uniform[0]).long()
    domain['X2']     = torch.from_numpy(bcl_u_pts_idx[0]).long()
    domain['Y1']     = torch.from_numpy(top_idx[0]).long()
    domain['Y2']     = torch.from_numpy(bottom_idx[0]).long()
    domain['Z1']     = torch.from_numpy(front_idx[0]).long()
    domain['Z2']     = torch.from_numpy(back_idx[0]).long()
    
    return domain

class S_Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out , act_fn):
        super(S_Net, self).__init__()
        self.act_fn = act_fn

        # self.encoding = rff.layers.GaussianEncoding(sigma=0.05, input_size=D_in, encoded_size=H//2)
        # self.encoding = rff.layers.PositionalEncoding(sigma=0.25, m=10)
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 2*H)
        self.linear3 = torch.nn.Linear(2*H, 4*H)
        self.linear4 = torch.nn.Linear(4*H, 2*H)
        self.linear5 = torch.nn.Linear(2*H, H)

        self.linear6 = torch.nn.Linear(H, D_out)
        
    def forward(self, x ):
        af_mapping = { 'tanh' : torch.tanh ,
                        'relu' : torch.nn.ReLU() ,
                        'rrelu' : torch.nn.RReLU() ,
                        'sigmoid' : torch.sigmoid }
        activation_fn = af_mapping[ self.act_fn ]  
          
        
        # y = self.encoding(x)
        y = activation_fn(self.linear1(x))
        y = activation_fn(self.linear2(y))
        y = activation_fn(self.linear3(y))
        y = activation_fn(self.linear4(y))
        y = activation_fn(self.linear5(y))

        # Output
        y = self.linear6(y)
        return y
    
    def reset_parameters(self):
        for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                    torch.nn.init.normal_(m.bias, mean=0, std=0.1)
                    
def loss_sum(tinput):
    return torch.sum(tinput) / tinput.data.nelement()
    
def innerproduct(A,B):
    Z = (A[:,0,0] * B[:,0,0] + A[:,0,1] * B[:,0,1] + A[:,0,2] * B[:,0,2] +
         A[:,1,0] * B[:,1,0] + A[:,1,1] * B[:,1,1] + A[:,1,2] * B[:,1,2] +
         A[:,2,0] * B[:,2,0] + A[:,2,1] * B[:,2,1] + A[:,2,2] * B[:,2,2])
         
    return Z
    
def determinant(F):

    detF = (F[:,0,0] * (F[:,1,1] * F[:,2,2] - F[:,1,2] * F[:,2,1])) - (
            F[:,0,1] * (F[:,1,0] * F[:,2,2] - F[:,1,2] * F[:,2,0])) + (
            F[:,0,2] * (F[:,1,0] * F[:,2,1] - F[:,1,1] * F[:,2,0]))
    
    return detF
    
def inverse(F):
    
    detF = determinant(F)
    F_inv = torch.empty((len(F),3,3))              
    F_inv[:,0,0] =  (F[:,1,1] * F[:,2,2] - F[:,1,2] * F[:,2,1]) / detF
    F_inv[:,0,1] = -(F[:,0,1] * F[:,2,2] - F[:,0,2] * F[:,2,1]) / detF
    F_inv[:,0,2] =  (F[:,0,1] * F[:,1,2] - F[:,0,2] * F[:,1,1]) / detF
    F_inv[:,1,0] = -(F[:,1,0] * F[:,2,2] - F[:,1,2] * F[:,2,0]) / detF
    F_inv[:,1,1] =  (F[:,0,0] * F[:,2,2] - F[:,0,2] * F[:,2,0]) / detF
    F_inv[:,1,2] = -(F[:,0,0] * F[:,1,2] - F[:,0,2] * F[:,1,0]) / detF
    F_inv[:,2,0] =  (F[:,1,0] * F[:,2,1] - F[:,1,1] * F[:,2,0]) / detF
    F_inv[:,2,1] = -(F[:,0,0] * F[:,2,1] - F[:,0,1] * F[:,2,0]) / detF
    F_inv[:,2,2] =  (F[:,0,0] * F[:,1,1] - F[:,0,1] * F[:,1,0]) / detF
    
    return F_inv

def trace(A):

    trace_A = A[:,0,0] + A[:,1,1] + A[:,2,2]

    return trace_A

def displacement_gradient(u,x):

    gradu = torch.empty((len(x),3,3))
    
    duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    
    du11 = duxdxyz[:, 0].unsqueeze(1); du12 = duxdxyz[:, 1].unsqueeze(1); du13 = duxdxyz[:, 2].unsqueeze(1)
    du21 = duydxyz[:, 0].unsqueeze(1); du22 = duydxyz[:, 1].unsqueeze(1); du23 = duydxyz[:, 2].unsqueeze(1)
    du31 = duzdxyz[:, 0].unsqueeze(1); du32 = duzdxyz[:, 1].unsqueeze(1); du33 = duzdxyz[:, 2].unsqueeze(1)
    
    gradu[:,0,0] = du11.squeeze(1); gradu[:,0,1] = du12.squeeze(1); gradu[:,0,2] = du13.squeeze(1)
    gradu[:,1,0] = du21.squeeze(1); gradu[:,1,1] = du22.squeeze(1); gradu[:,1,2] = du23.squeeze(1)
    gradu[:,2,0] = du31.squeeze(1); gradu[:,2,1] = du32.squeeze(1); gradu[:,2,2] = du33.squeeze(1)

    # For diagonal case
    # gradu[:,0,1]=0; gradu[:,0,2]=0; gradu[:,1,0]=0; gradu[:,1,2]=0; gradu[:,2,0]=0; gradu[:,2,1]=0

    return gradu

def deformation_gradient(u,x):

    identity = torch.zeros((len(x), 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1
    gradu    = displacement_gradient(u, x)
    F        = identity + gradu

    return F
    
def stressNH(F):
    # Material Properties
    lmbdaNH   = YM * PR /(1+PR)/(1-2*PR)
    muNH      = YM/2/(1+PR)
    Finv     = inverse(F)
    detF     = determinant(F)
    stressPK = muNH * F + (lmbdaNH * torch.log(detF) - muNH).view(-1,1,1)*Finv.permute(0,2,1)

    return stressPK


def psi(u_pred, x, integrationIE, dx, dy, dz, shape):
    mu = YM / ( 2. * ( 1. + PR ) )
    K = YM / ( 3. * ( 1. - 2. * PR ) )
    C10 = mu / 2.
    D1 = 2. / K
    # print( C10 , D1 )

    F    = deformation_gradient(u_pred, x)
    detF = determinant(F)
    F_bar = torch.einsum( 'i,ijk->ijk' , torch.pow( detF , -1./3. ) , F )
    B_bar    = torch.bmm( F_bar , F_bar.permute(0,2,1) )
    I1   = trace( B_bar )

    psiE = C10 * ( I1 - 3. ) + torch.pow( detF - 1 , 2. ) / D1

    internal_1 = integrationIE(psiE, dx=dx, dy=dy, dz=dz, shape=[shape[0], shape[1], shape[2]])
    
    return internal_1


def psi_Gauss(u, x, integrationIE, dx, dy, dz, shape):
    mu = YM / ( 2. * ( 1. + PR ) )
    K = YM / ( 3. * ( 1. - 2. * PR ) )
    C10 = mu / 2.
    D1 = 2. / K

    N_element = ( shape[0] - 1 ) * ( shape[1] - 1 ) * ( shape[2] - 1 )
    order = [ 1 ,  shape[-1] , shape[0] , shape[1] ]
    Ux = torch.transpose(u[:, 0].reshape( order ), 2, 3)
    Uy = torch.transpose(u[:, 1].reshape( order ), 2, 3)
    Uz = torch.transpose(u[:, 2].reshape( order ), 2, 3)
    U = torch.cat( (Ux,Uy,Uz) , dim=0 )

    #        dim  z      y     x
    U_N1 = U[ : , :-1 , :-1 , :-1 ]
    U_N2 = U[ : , :-1 , :-1 , 1: ]
    U_N3 = U[ : , 1: , :-1 , 1: ]
    U_N4 = U[ : , 1: , :-1 , :-1 ]
    U_N5 = U[ : , :-1 , 1: , :-1 ]
    U_N6 = U[ : , :-1 , 1: , 1: ]
    U_N7 = U[ : , 1: , 1: , 1: ]
    U_N8 = U[ : , 1: , 1: , :-1 ]
    U_N = torch.stack( [ U_N1 , U_N2 , U_N3 , U_N4 , U_N5 , U_N6 , U_N7 , U_N8 ] )#.double()

    # Compute constants
    detJ = dx*dy*dz / 8.
    Jinv = torch.zeros([3,3]).double()
    dxdydz = [ dx , dy , dz ]
    for i in range(3):
        Jinv[i,i] = 2. / dxdydz[i]
    identity = torch.zeros((N_element, 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1


    # Go through all integration pts
    strainEnergy_at_elem = torch.zeros( N_element )

    vv = np.sqrt( 1. / 3. )
    pt = [-vv,vv]
    intpt = torch.tensor([[pt[0],pt[0],pt[0]],
                          [pt[1],pt[0],pt[0]],
                          [pt[1],pt[1],pt[0]],
                          [pt[0],pt[1],pt[0]],
                          [pt[0],pt[0],pt[1]],
                          [pt[1],pt[0],pt[1]],
                          [pt[1],pt[1],pt[1]],
                          [pt[0],pt[1],pt[1]]])

    for i in range( 8 ):
        x_ , y_ , z_ = intpt[i,:]
        # Shape grad in natural coords
        B = torch.tensor([[-((y_ - 1)*(z_ - 1))/8, -((x_ - 1)*(z_ - 1))/8, -((x_ - 1)*(y_ - 1))/8],
                    [ ((y_ - 1)*(z_ - 1))/8,  ((x_ + 1)*(z_ - 1))/8,  ((x_ + 1)*(y_ - 1))/8],
                    [-((y_ - 1)*(z_ + 1))/8, -((x_ + 1)*(z_ + 1))/8, -((x_ + 1)*(y_ - 1))/8],
                    [ ((y_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(y_ - 1))/8],
                    [ ((y_ + 1)*(z_ - 1))/8,  ((x_ - 1)*(z_ - 1))/8,  ((x_ - 1)*(y_ + 1))/8],
                    [-((y_ + 1)*(z_ - 1))/8, -((x_ + 1)*(z_ - 1))/8, -((x_ + 1)*(y_ + 1))/8],
                    [ ((y_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(y_ + 1))/8],
                    [-((y_ + 1)*(z_ + 1))/8, -((x_ - 1)*(z_ + 1))/8, -((x_ - 1)*(y_ + 1))/8]]).double()
        
        # Convert to physical gradient
        B_physical = torch.matmul( B , Jinv ).double()
        dUx = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,0,:,:,:] , B_physical )
        dUy = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,1,:,:,:] , B_physical )
        dUz = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,2,:,:,:] , B_physical )
        grad_u = torch.reshape( torch.transpose( torch.flatten( torch.cat( (dUx,dUy,dUz) , dim=0 ) ,  start_dim=1, end_dim=-1 ) , 0 , 1 ) , [N_element,3,3] )

        # Def grad
        F = grad_u + identity

        detF = determinant( F )
        F_bar = torch.einsum( 'i,ijk->ijk' , torch.pow( detF , -1./3. ) , F )
        B_bar    = torch.bmm( F_bar , F_bar.permute(0,2,1) )
        I1   = trace( B_bar )

        psiE = C10 * ( I1 - 3. ) + torch.pow( detF - 1 , 2. ) / D1

        strainEnergy_at_elem += psiE * 1. * detJ    
    return torch.sum( strainEnergy_at_elem )


def stressLE( e ):
    lame1 = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) )
    mu = YM / ( 2. * ( 1. + PR ) )    

    identity = torch.zeros((len(e), 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1

    trace_e = e[:,0,0] + e[:,1,1] + e[:,2,2]
    return lame1 * torch.einsum( 'ijk,i->ijk' , identity , trace_e ) + 2 * mu * e


def LE(u_pred, x, integrationIE, dx, dy, dz, shape):
    grad_u    = displacement_gradient(u_pred, x)
    strain = 0.5 * ( grad_u + grad_u.permute(0,2,1) )

    stress = stressLE( strain )

    psiE = 0.5 * torch.einsum( 'ijk,ijk->i' , stress , strain )

    internal_1 = integrationIE(psiE, dx=dx, dy=dy, dz=dz, shape=[shape[0], shape[1], shape[2]])
    
    return internal_1


def LE_Gauss(u, x, integrationIE, dx, dy, dz, shape):
    N_element = ( shape[0] - 1 ) * ( shape[1] - 1 ) * ( shape[2] - 1 )
    order = [ 1 ,  shape[-1] , shape[0] , shape[1] ]
    Ux = torch.transpose(u[:, 0].reshape( order ), 2, 3)
    Uy = torch.transpose(u[:, 1].reshape( order ), 2, 3)
    Uz = torch.transpose(u[:, 2].reshape( order ), 2, 3)
    U = torch.cat( (Ux,Uy,Uz) , dim=0 )

    #        dim  z      y     x
    U_N1 = U[ : , :-1 , :-1 , :-1 ]
    U_N2 = U[ : , :-1 , :-1 , 1: ]
    U_N3 = U[ : , 1: , :-1 , 1: ]
    U_N4 = U[ : , 1: , :-1 , :-1 ]
    U_N5 = U[ : , :-1 , 1: , :-1 ]
    U_N6 = U[ : , :-1 , 1: , 1: ]
    U_N7 = U[ : , 1: , 1: , 1: ]
    U_N8 = U[ : , 1: , 1: , :-1 ]
    U_N = torch.stack( [ U_N1 , U_N2 , U_N3 , U_N4 , U_N5 , U_N6 , U_N7 , U_N8 ] )#.double()

    # Compute constants
    detJ = dx*dy*dz / 8.
    Jinv = torch.zeros([3,3]).double()
    dxdydz = [ dx , dy , dz ]
    for i in range(3):
        Jinv[i,i] = 2. / dxdydz[i]

    grad2strain = torch.zeros([6,9]).double()
    grad2strain[0,0] = 1. # 11
    grad2strain[1,4] = 1. # 22
    grad2strain[2,8] = 1. # 33
    grad2strain[3,5] = 0.5; grad2strain[3,7] = 0.5 # 23
    grad2strain[4,2] = 0.5; grad2strain[4,6] = 0.5 # 13
    grad2strain[5,1] = 0.5; grad2strain[5,3] = 0.5 # 12 

    C_elastic = torch.zeros([6,6]).double()
    C_elastic[0,0] = 1. - PR; C_elastic[0,1] = PR; C_elastic[0,2] = PR
    C_elastic[1,0] = PR; C_elastic[1,1] = 1. - PR; C_elastic[1,2] = PR
    C_elastic[2,0] = PR; C_elastic[2,1] = PR; C_elastic[2,2] = 1. - PR
    C_elastic[3,3] = 1. - 2. * PR;
    C_elastic[4,4] = 1. - 2. * PR;
    C_elastic[5,5] = 1. - 2. * PR;
    C_elastic *= ( YM / ( ( 1. + PR ) * ( 1. - 2. * PR ) ) )

    # Go through all integration pts
    strainEnergy_at_elem = torch.zeros( [ shape[-1] -1 , shape[1] -1 , shape[0] -1 ] )

    vv = np.sqrt( 1. / 3. )
    pt = [-vv,vv]
    intpt = torch.tensor([[pt[0],pt[0],pt[0]],
                          [pt[1],pt[0],pt[0]],
                          [pt[1],pt[1],pt[0]],
                          [pt[0],pt[1],pt[0]],
                          [pt[0],pt[0],pt[1]],
                          [pt[1],pt[0],pt[1]],
                          [pt[1],pt[1],pt[1]],
                          [pt[0],pt[1],pt[1]]])

    for i in range( 8 ):
        x_ , y_ , z_ = intpt[i,:]
        # Shape grad in natural coords
        B = torch.tensor([[-((y_ - 1)*(z_ - 1))/8, -((x_ - 1)*(z_ - 1))/8, -((x_ - 1)*(y_ - 1))/8],
                    [ ((y_ - 1)*(z_ - 1))/8,  ((x_ + 1)*(z_ - 1))/8,  ((x_ + 1)*(y_ - 1))/8],
                    [-((y_ - 1)*(z_ + 1))/8, -((x_ + 1)*(z_ + 1))/8, -((x_ + 1)*(y_ - 1))/8],
                    [ ((y_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(y_ - 1))/8],
                    [ ((y_ + 1)*(z_ - 1))/8,  ((x_ - 1)*(z_ - 1))/8,  ((x_ - 1)*(y_ + 1))/8],
                    [-((y_ + 1)*(z_ - 1))/8, -((x_ + 1)*(z_ - 1))/8, -((x_ + 1)*(y_ + 1))/8],
                    [ ((y_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(y_ + 1))/8],
                    [-((y_ + 1)*(z_ + 1))/8, -((x_ - 1)*(z_ + 1))/8, -((x_ - 1)*(y_ + 1))/8]]).double()
        
        # Convert to physical gradient
        B_physical = torch.matmul( B , Jinv ).double()
        dUx = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,0,:,:,:] , B_physical )
        dUy = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,1,:,:,:] , B_physical )
        dUz = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,2,:,:,:] , B_physical )
        dU = torch.cat( (dUx,dUy,dUz) , dim=0 )

        # Strain [ 11 , 22 , 33 , 23 , 13 , 12 ]
        eps = torch.einsum( 'qi,ijkl->qjkl' , grad2strain , dU )

        # Stress [ 11 , 22 , 33 , 23 , 13 , 12 ]
        Cauchy = torch.einsum( 'qi,ijkl->qjkl' , C_elastic , eps )

        # Shear stresses need to be counted twice due to symmetry
        Cauchy[3:,:,:,:] *= 2.
        SE = 0.5 * torch.einsum( 'ijkl,ijkl->jkl' , Cauchy , eps ) 

        # Scaled by design density
        strainEnergy_at_elem += SE * 1. * detJ    
    return torch.sum( strainEnergy_at_elem )


def CauchyStress(P, F):

    detF  = determinant(F)
    sigma = torch.pow(detF,-1).view(-1,1,1) * torch.bmm(P,F.permute(0,2,1)) 
    return sigma

def strain(F):

    identity = torch.zeros((len(F), 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1
    C        = torch.bmm(F.permute(0,2,1),F)
    strainCG = 0.5 * (C-identity)
    
    return strainCG

def ConvergenceCheck( arry , rel_tol ):
    num_check = 10

    if HyperOPT and arry[-1] < -4.:
        print('Solution diverged!!!!!!!')
        return True

    # Run minimum of 2*num_check iterations
    if len( arry ) < 2 * num_check :
        return False

    mean1 = np.mean( arry[ -2*num_check : -num_check ] )
    mean2 = np.mean( arry[ -num_check : ] )

    if np.abs( mean2 ) < 1e-6:
        print('Loss value converged to abs tol of 1e-6' )
        return True     

    if ( np.abs( mean1 - mean2 ) / np.abs( mean2 ) ) < rel_tol:
        print('Loss value converged to rel tol of ' + str(rel_tol) )
        return True
    else:
        return False


class DeepMixedMethod:
    # Instance attributes
    def __init__(self, model):
        self.S_Net   = S_Net(model[0], model[1], model[2] , model[4] )
        self.S_Net   = self.S_Net.to(dev)
        numIntType   = 'AD'# 'AD'  'trapezoidal'
        self.intLoss = IntegrationLoss(numIntType, 3)
        self.lr = model[3]


    def train_model(self, domain):
        N_para = 0
        for parameter in self.S_Net.parameters():
            N_para += np.sum( list(parameter.shape) )
        # print( N_para )
        # exit()
                
        integrationIE = self.intLoss.lossInternalEnergy
        integrationEE = self.intLoss.lossExternalEnergy
        nodes   = domain['Domain'].to(dev); nodes.requires_grad_(True);   nodes.retain_grad()
        nodesEn = domain['Energy'].to(dev); nodesEn.requires_grad_(True); nodesEn.retain_grad()
        X1_indx = domain['X1'].to(dev); X2_indx = domain['X2'].to(dev)
        Y1_indx = domain['Y1'].to(dev); Y2_indx = domain['Y2'].to(dev)
        Z1_indx = domain['Z1'].to(dev); Z2_indx = domain['Z2'].to(dev)
        node_int_indx = domain['Xint'].to(dev)
        node_X1_Uni_indx = domain['X1_Uni'].to(dev)
        
        X1    = nodes[X1_indx,:]; X1.requires_grad_(True); X1.retain_grad()
        X2    = nodes[X2_indx,:]; X2.requires_grad_(True); X2.retain_grad()
        Y1    = nodes[Y1_indx,:]; Y1.requires_grad_(True); Y1.retain_grad()
        Y2    = nodes[Y2_indx,:]; Y2.requires_grad_(True); Y2.retain_grad()
        Z1    = nodes[Z1_indx,:]; Z1.requires_grad_(True); Z1.retain_grad()
        Z2    = nodes[Z2_indx,:]; Z2.requires_grad_(True); Z2.retain_grad()
        nodes_int = nodes[node_int_indx,:]; nodes_int.requires_grad_(True); nodes_int.retain_grad()
        X1_Uni = nodes[node_X1_Uni_indx,:]; X1_Uni.requires_grad_(True); X1_Uni.retain_grad()
        
        identity = torch.zeros((len(nodes), 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1
        torch.set_printoptions(precision=8)
        self.S_Net.reset_parameters()

        
        LBFGS_max_iter  = 200
        optimizerL = torch.optim.LBFGS(self.S_Net.parameters(), lr=self.lr, max_iter=LBFGS_max_iter, line_search_fn='strong_wolfe', tolerance_change=1e-8, tolerance_grad=1e-8)
        LBFGS_loss = {}

        disp_history     = np.zeros((step_max+1,len(nodes), 3))
        F_histroy        = np.zeros((step_max+1,len(nodes), 3, 3))
        strainCG_histroy = np.zeros((step_max+1,len(nodes), 3, 3))
        stressPK_histroy = np.zeros((step_max+1,len(nodes), 3, 3))
        stressC_histroy = np.zeros((step_max+1,len(nodes), 3, 3))
        electric_potential_M2 = torch.zeros( len(nodes) )
        E_field = torch.zeros((len(nodes), 3))

        for step in range(1,step_max+1):
            self.applied_trac = step/step_max * total_traction
            # print(self.applied_trac)
                
            tempL = []
            for epoch in range(LBFGS_Iteration):
                def closure():
                    loss = self.loss_function(step,epoch,nodes,nodes_int,X1_Uni,X1,X2,Y1,Y2,Z1,Z2,nodesEn,self.applied_trac, integrationIE, integrationEE)
                    optimizerL.zero_grad()
                    loss.backward(retain_graph=True)
                    tempL.append(loss.item())
                    return loss
                optimizerL.step(closure)

                # Check convergence
                if ConvergenceCheck( tempL , rel_tol ):
                    break

            LBFGS_loss[step] = tempL

            u_pred = self.getUP(nodes)          
            F_M2        = deformation_gradient(u_pred, nodes)

            if Example == 'Hyperelastic':        
                F_M2        = deformation_gradient(u_pred, nodes)
                strainCG_M2 = strain(F_M2)
                stressPK_M2 = stressNH(F_M2)
                stressC_M2 = CauchyStress(stressPK_M2, F_M2)

            elif Example == 'Elastic':
                grad_u    = displacement_gradient(u_pred, nodes)
                strainCG_M2 = 0.5 * ( grad_u + grad_u.permute(0,2,1) )
                stressC_M2 = stressLE( strainCG_M2 )

            elif Example == 'Piezoelectric':
                grad_u    = displacement_gradient(u_pred, nodes)
                E_field    = electric_potential_gradient(u_pred, nodes)

                strainCG_M2 = 0.5 * ( grad_u + grad_u.permute(0,2,1) )

                stressC_M2 = stressLE( strainCG_M2 ) - torch.einsum( 'jkl,il->ijk' , e , E_field )

                electric_potential_M2 = u_pred[:,3]



            disp_history[step,:, :]        = u_pred[:,:3].detach().cpu().numpy()
            strainCG_histroy[step,:, :, :] = strainCG_M2.detach().cpu().numpy()
            stressC_histroy[step,:, :, :]  = stressC_M2.detach().cpu().numpy()
                
        return disp_history, F_histroy, strainCG_histroy, stressPK_histroy, stressC_histroy, stressC_M2, strainCG_M2 , electric_potential_M2 , E_field , LBFGS_loss

    def getUP(self, nodes):
        uP  = self.S_Net.forward(nodes).double()
        Ux = nodes[:, 0] * uP[:, 0]
        Uy = nodes[:, 0] * uP[:, 1]
        Uz = nodes[:, 0] * uP[:, 2]
        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        Uz = Uz.reshape(Uz.shape[0], 1)
        u_pred = torch.cat((Ux, Uy, Uz), -1)
        return u_pred


    def loss_function(self,step,epoch,nodes,nodes_int,X1_Uni,X1,X2,Y1,Y2,Z1,Z2,nodesEn,traction,integrationIE, integrationEE):
        u_nodesE = self.getUP(nodesEn)
        if Example == 'Hyperelastic':
            if INT_TYPE == 'AD':
                internal = psi(u_nodesE, nodesEn, integrationIE, dx, dy, dz, shape)
            else:
                internal = psi_Gauss(u_nodesE, nodesEn, integrationIE, dx, dy, dz, shape)

        elif Example == 'Elastic':
            if INT_TYPE == 'AD':
                internal = LE(u_nodesE, nodesEn, integrationIE, dx, dy, dz, shape)
            else:
                internal = LE_Gauss(u_nodesE, nodesEn, integrationIE, dx, dy, dz, shape)


        neu_uP_pred = self.getUP(X1_Uni)[:,:3]
        neu_u_pred = neu_uP_pred[:,1]
        fext = traction * neu_u_pred 
        external = integrationEE(fext, dx=dy, dy=dz, shape=[shape[1], shape[2]])
        L_E = internal - external
        loss =  L_E
        print('Step = '+ str(step) + ', Epoch = ' + str( epoch) + ', L = ' + str( loss.item() ) )        
        return loss
        

global Example
Example = 'Hyperelastic'
# Example = 'Elastic'

INT_TYPE = 'AD'
# INT_TYPE = 'SF'


print('Example = ' + Example + ', using ' + INT_TYPE )
base  = './DEM/' + Example + '_' + INT_TYPE + '/'
if not os.path.exists( base ):
    os.mkdir( base )

# ------------------------------ network settings ---------------------------------------------------
D_in  = 3
H     = 16
D_out = 3

# ----------------------------- define structural parameters ----------------------------------------
Length = 4.0
Width  = 1.0
Depth  = 1.0

numb_nodes_cont_param = 10
Ny = numb_nodes_cont_param
Nz = numb_nodes_cont_param
Nx = int((numb_nodes_cont_param-1) * int(Length/Width) + 1)

# Nx = 67; Ny = 18; Nz = 18


shape = [Nx, Ny, Nz]
x_min, y_min, z_min = (0.0, 0.0, 0.0)
(dx, dy, dz) = (Length / (Nx - 1), Width / (Ny - 1), Depth / (Nz - 1))

domain = setup_domain()
print('# of nodes is ', len(domain['Domain']))
print('# of interior nodes is ', len(domain['Xint']))
print('# of surfaace nodes is ', len(domain['Domain']) - len(domain['Xint']))


YM =  1000
PR =  0.3



# Loading
total_traction = -25.
step_max   = 20
ref_file = './AbaqusReferenceDisplacements/' + 'NH_Disp25_'


# Training
LBFGS_Iteration = 20
rel_tol = 5e-5



# Initial hyper parameters
x_var = { 'x_lr' : 0.01 ,
         'neuron' : 16 ,
         'act_func' : 'tanh' }

def Obj( x_var ):
    lr = x_var['x_lr']
    H = int(x_var['neuron'])
    act_fn = x_var['act_func']
    print( 'LR: ' + str(lr) + ', H: ' + str(H) + ', act fn: ' + act_fn )


    dcm = DeepMixedMethod([D_in, H, D_out, lr , act_fn])
    start_time = time.time()
    disp_history, F_histroy, strainCG_histroy, stressPK_histroy, stressC_history, stressC_last, strain_last , electric_potential_last , E_field , LBFGS_loss = dcm.train_model(domain)
    end_time = time.time()
    print('simulation time = ' + str(end_time - start_time) + 's')



    #######################################################################################################################################
    # Save data
    x_space = np.expand_dims(domain['Domain'][:,0].detach().cpu().numpy(), axis=1)
    y_space = np.expand_dims(domain['Domain'][:,1].detach().cpu().numpy(), axis=1)
    z_space = np.expand_dims(domain['Domain'][:,2].detach().cpu().numpy(), axis=1)
    coordin = np.concatenate((x_space, y_space, z_space), axis=1)
    U = disp_history[-1,:,:]

    Nodal_Strain = torch.cat((strain_last[:,0,0].unsqueeze(1),strain_last[:,1,1].unsqueeze(1),strain_last[:,2,2].unsqueeze(1),\
                              strain_last[:,0,1].unsqueeze(1),strain_last[:,1,2].unsqueeze(1),strain_last[:,0,2].unsqueeze(1)),axis=1)
    Nodal_Stress = torch.cat((stressC_last[:,0,0].unsqueeze(1),stressC_last[:,1,1].unsqueeze(1),stressC_last[:,2,2].unsqueeze(1),\
                              stressC_last[:,0,1].unsqueeze(1),stressC_last[:,1,2].unsqueeze(1),stressC_last[:,0,2].unsqueeze(1)),axis=1)
    Nodal_E = torch.cat((E_field[:,0].unsqueeze(1),E_field[:,1].unsqueeze(1),E_field[:,2].unsqueeze(1)),axis=1)

    stress_vMis = torch.pow(0.5 * (torch.pow((Nodal_Stress[:,0]-Nodal_Stress[:,1]), 2) + torch.pow((Nodal_Stress[:,1]-Nodal_Stress[:,2]), 2)
                   + torch.pow((Nodal_Stress[:,2]-Nodal_Stress[:,0]), 2) + 6 * (torch.pow(Nodal_Stress[:,3], 2) +
                     torch.pow(Nodal_Stress[:,4], 2) + torch.pow(Nodal_Stress[:,5], 2))), 0.5)
    Nodal_Strain = Nodal_Strain.cpu().detach().numpy()
    Nodal_Stress = Nodal_Stress.cpu().detach().numpy()
    stress_vMis  = stress_vMis.unsqueeze(1).cpu().detach().numpy()
    electric_potential = electric_potential_last.unsqueeze(1).cpu().detach().numpy()
    Nodal_E = Nodal_E.cpu().detach().numpy()




    Data = np.concatenate((coordin, U, Nodal_Strain , Nodal_Stress, stress_vMis, electric_potential , Nodal_E), axis=1)
    np.save( base + 'Results.npy',Data)

    LBFGS_loss_D1 = np.array(LBFGS_loss[1])
    fn = base + 'Training_loss.npy'
    np.save( fn , LBFGS_loss_D1 )



    #######################################################################################################################################
    # Write vtk
    def FormatMe( v ):
        S = [Nz,Nx,Ny]
        return np.swapaxes( np.swapaxes( v.reshape(S) , 0 , 1 ) , 1 , 2 ).flatten('F')

    grid = pv.UniformGrid()
    grid.dimensions = np.array([Nx,Ny,Nz])
    grid.origin = np.zeros(3)
    grid.spacing = np.array([dx,dy,dz])
    names = [ 'Ux' , 'Uy' , 'Uz' , 'E11' , 'E22' , 'E33' , 'E12' , 'E23' , 'E13' , 'S11' , 'S22' , 'S33' , 'S12' , 'S23' , 'S13' , 'SvM' , 'E_pot' , 'D1' , 'D2' , 'D3']
    for idx , n in enumerate( names ):
        grid.point_data[ n ] =  FormatMe( Data[:,idx+3] )



    #############################################################################################
    # Abaqus comparison
    Out = np.load( ref_file + '.npy' )

    names = [ 'Ux_ABQ' , 'Uy_ABQ' , 'Uz_ABQ' ]
    for idx , n in enumerate( names ):
        grid.point_data[ n ] =  Out[idx].flatten('F')

    # Compute difference
    names = [ 'Ux' , 'Uy' , 'Uz' ]
    diff = []
    for idx , n in enumerate( names ):
        FEM = grid.point_data[ n + '_ABQ' ]
        ML = grid.point_data[ n ]
        grid.point_data[ n + '_diff' ] =  np.abs( FEM - ML ) / np.max( np.abs(FEM) ) * 100.
        diff.append( np.mean(grid.point_data[ n + '_diff' ]) )
    grid.save( base + "Results.vti")

    mE = np.mean(diff)
    print( 'Mean error in U compared to Abaqus: ' + str(mE) )
    return np.mean(diff)

Obj( x_var )