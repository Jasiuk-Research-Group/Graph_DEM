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
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, TransformerConv, TAGConv, ARMAConv, SGConv, MFConv, RGCNConv
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from pprint import pprint
import pyvista as pv
torch.manual_seed(2022)
mpl.rcParams['figure.dpi'] = 350


torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    device = torch.device('cuda')
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
    # np.meshgrid(lin_x, lin_y, lin_z)

    dom = torch.from_numpy(domEn).float()
    t1 = time.time()
    G = create_graph(dom)
    print( 'Building graph took ' + str(time.time()-t1) + ' s' )

    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx = np.where(dom[:, 0] == 0)
    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where(dom[:, 0] == Length)
    top_idx = np.where((dom[:, 1]==Width) & (dom[:, 0]>0) & (dom[:, 0]<Length))
    bottom_idx = np.where((dom[:, 1] == 0) & (dom[:, 0]>0) & (dom[:, 0]<Length))
    front_idx = np.where((dom[:, 2] == Depth) & (dom[:, 0]>0) & (dom[:, 0]<Length))
    back_idx = np.where((dom[:, 2] == 0) & (dom[:, 0]>0) & (dom[:, 0]<Length))

    Boundaries           = {}
    Boundaries['X1']     = torch.from_numpy(bcr_t_pts_idx[0]).long()
    Boundaries['X2']     = torch.from_numpy(bcl_u_pts_idx[0]).long()
    Boundaries['Y1']     = torch.from_numpy(top_idx[0]).long()
    Boundaries['Y2']     = torch.from_numpy(bottom_idx[0]).long()
    Boundaries['Z1']     = torch.from_numpy(front_idx[0]).long()
    Boundaries['Z2']     = torch.from_numpy(back_idx[0]).long()
    
    return G, Boundaries

def create_graph(coordinates):
    A = np.array(coordinates)
    B = squareform(pdist(A))
    # print("B=", B)
    G = nx.from_numpy_matrix(B)
    for i in range(len(coordinates)):
        G.add_node(i, coordinates=coordinates[i])
        for j in range(len(coordinates)):
            if G.get_edge_data(i,j) !=None:
                if (G.get_edge_data(i,j)['weight']>dx):
                    # print("i=",i, "j=",j)
                    G.remove_edge(i,j)
    # analyze_graph(G)
    # pprint(vars(G))

    return G


def analyze_graph(G):
    nx.draw(G)
    print("nodes=", G.number_of_nodes(), "edges=", G.number_of_edges())
    plt.show()
    
# custom dataset
class MeshDataSet(InMemoryDataset):
    def __init__(self, transform=None):
        super(MeshDataSet, self).__init__('.', transform, None, None)

        data = Data(edge_index=edge_index)
        
        data.num_nodes = G.number_of_nodes()
        
        # Using degree as embedding
        embeddings = np.zeros((G.number_of_nodes(),3))
        for i in range(G.number_of_nodes()):
            embeddings[i,:] = G.nodes[i]['coordinates']
        data.nodes = torch.from_numpy(embeddings).float()
        # normalizing degree values
        scale = StandardScaler()
        embeddings = scale.fit_transform(embeddings.reshape(-1,3))
        
        # embedding 
        data.x = torch.from_numpy(embeddings).float()
        

        data.num_classes = 3


        n_nodes = G.number_of_nodes()
        
        # create train and test masks for data
        X_train = pd.Series(list(G.nodes()))
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        data['train_mask'] = train_mask

        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    

# GCN model 
class GCNet(torch.nn.Module):
    def __init__(self,  D_in, H, D_out , act_fn ):
        super(GCNet, self).__init__()
        self.act_fn = act_fn
        
        # self.conv1 = GCNConv(D_in, H)
        # self.conv2 = GCNConv(H, 2*H)
        # self.conv3 = GCNConv(2*H, 4*H)
        # self.conv4 = GCNConv(4*H, 2*H)
        # self.conv5 = GCNConv(2*H, H)
        # self.conv6 = GCNConv(H, D_out)
        
        kk=1
        self.conv1 = ChebConv(D_in, H, K=kk)
        self.conv2 = ChebConv(H, 2*H, K=kk)
        self.conv3 = ChebConv(2*H, 4*H, K=kk)
        self.conv4 = ChebConv(4*H, 2*H, K=kk)
        self.conv5 = ChebConv(2*H, H, K=kk)
        # self.conv6 = ChebConv(H, D_out, K=kk)
        
        # self.conv1 = GATConv(D_in, H)
        # self.conv2 = GATConv(H, 2*H)
        # self.conv3 = GATConv(2*H, 4*H)
        # self.conv4 = GATConv(4*H, 2*H)
        # self.conv5 = GATConv(2*H, H)
        # self.conv6 = GATConv(H, D_out)
        
        # self.conv1 = TransformerConv(D_in, H)
        # self.conv2 = TransformerConv(H, 2*H)
        # self.conv3 = TransformerConv(2*H, 4*H)
        # self.conv4 = TransformerConv(4*H, 2*H)
        # self.conv5 = TransformerConv(2*H, H)
        # self.conv6 = TransformerConv(H, D_out)
        
        # self.conv1 = TAGConv(D_in, H)
        # self.conv2 = TAGConv(H, 2*H)
        # self.conv3 = TAGConv(2*H, 4*H)
        # self.conv4 = TAGConv(4*H, 2*H)
        # self.conv5 = TAGConv(2*H, H)
        # self.conv6 = TAGConv(H, D_out)
        
        # self.conv1 = ARMAConv(D_in, H)
        # self.conv2 = ARMAConv(H, 2*H)
        # self.conv3 = ARMAConv(2*H, 4*H)
        # self.conv4 = ARMAConv(4*H, 2*H)
        # self.conv5 = ARMAConv(2*H, H)
        # self.conv6 = ARMAConv(H, D_out)
        
        # self.conv1 = SGConv(D_in, H)
        # self.conv2 = SGConv(H, 2*H)
        # self.conv3 = SGConv(2*H, 4*H)
        # self.conv4 = SGConv(4*H, 2*H)
        # self.conv5 = SGConv(2*H, H)
        # self.conv6 = SGConv(H, D_out)
        
        # self.conv1 = MFConv(D_in, H)
        # self.conv2 = MFConv(H, 2*H)
        # self.conv3 = MFConv(2*H, 4*H)
        # self.conv4 = MFConv(4*H, 2*H)
        # self.conv5 = MFConv(2*H, H)
        # self.conv6 = MFConv(H, D_out)
        
        self.linear1 = torch.nn.Linear(H,D_out)
        
    def forward(self, x_coord, edge_index ):
        af_mapping = { 'tanh' : torch.tanh ,
                        'relu' : torch.nn.ReLU() ,
                        'rrelu' : torch.nn.RReLU() ,
                        'sigmoid' : torch.sigmoid }
        activation_fn = af_mapping[ self.act_fn ]  
        
        y = self.conv1(x_coord, edge_index)
        y = activation_fn(y)
        y = self.conv2(y, edge_index)
        y = activation_fn(y)
        y = self.conv3(y, edge_index)
        y = activation_fn(y)
        y = self.conv4(y, edge_index)
        y = activation_fn(y)
        y = self.conv5(y, edge_index)
        y = activation_fn(y)

        # Output
        y = self.linear1(y)
        return y
                    
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
    
    duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=device), create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=device), create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=device), create_graph=True, retain_graph=True)[0]
    
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

def stressLE( e ):
    lame1 = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) )
    mu = YM / ( 2. * ( 1. + PR ) )    

    identity = torch.zeros((len(e), 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1

    trace_e = e[:,0,0] + e[:,1,1] + e[:,2,2]
    return lame1 * torch.einsum( 'ijk,i->ijk' , identity , trace_e ) + 2 * mu * e

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
        self.GCNet   = GCNet( model[0], model[1], model[2] , model[4] )
        self.GCNet   = self.GCNet.to(device)
        numIntType   = 'AD'# 'AD'  'trapezoidal'
        self.intLoss = IntegrationLoss(numIntType, 3)
        self.lr = model[3]

    def train_model(self):   
        integrationIE = self.intLoss.lossInternalEnergy
        integrationEE = self.intLoss.lossExternalEnergy
        torch.set_printoptions(precision=8)        
        x_scaled, edge_index = nodes.x, nodes.edge_index
        x_coord = nodes.nodes
        x_coord.requires_grad_(True);   x_coord.retain_grad()

        optimizerL = torch.optim.LBFGS(self.GCNet.parameters(), lr=self.lr, max_iter=200, line_search_fn='strong_wolfe', tolerance_change=1e-6, tolerance_grad=1e-6)
        
        LOSS = {}
        disp_history     = np.zeros((step_max+1,nodes.num_nodes, 3))
        strainCG_histroy = np.zeros((step_max+1,nodes.num_nodes, 3, 3))
        stressC_histroy = np.zeros((step_max+1,nodes.num_nodes, 3, 3))
        electric_potential_M2 = torch.zeros( nodes.num_nodes )
        E_field = torch.zeros((nodes.num_nodes, 3))

        for step in range(1,step_max+1):
            self.applied_trac = step/step_max * total_traction
            tempL = []
            for epoch in range(epochs):
                def closure():
                    loss = self.loss_function(step, epoch, x_scaled, edge_index, x_coord, X1_idx,self.applied_trac, integrationIE, integrationEE)
                    optimizerL.zero_grad()
                    loss.backward(retain_graph=True)
                    tempL.append(loss.item())
                    return loss
                optimizerL.step(closure)

                # Check convergence
                if ConvergenceCheck( tempL , rel_tol ):
                    break

            LOSS[step] = tempL
            
        
            u_pred      = self.getU(x_coord, edge_index)  
            disp_history[step,:, :]        = u_pred[:,:3].detach().cpu().numpy()

            if Example == 'Hyperelastic':        
                F_M2        = deformation_gradient(u_pred, x_coord)
                strainCG_M2 = strain(F_M2)
                stressPK_M2 = stressNH(F_M2)
                stressC_M2 = CauchyStress(stressPK_M2, F_M2)

            elif Example == 'Elastic':
                grad_u    = displacement_gradient(u_pred, x_coord)
                strainCG_M2 = 0.5 * ( grad_u + grad_u.permute(0,2,1) )
                stressC_M2 = stressLE( strainCG_M2 )

            stressC_histroy[step,:, :, :]  = stressC_M2.detach().cpu().numpy()
            strainCG_histroy[step,:, :, :] = strainCG_M2.detach().cpu().numpy()
                
        return disp_history, strainCG_histroy, stressC_histroy, stressC_M2, strainCG_M2 , electric_potential_M2 , E_field , LOSS

    def getU(self, x_coord, edge_index):
        u  = self.GCNet.forward(x_coord, edge_index).double()

        Ux = x_coord[:, 0] * u[:, 0]
        Uy = x_coord[:, 0] * u[:, 1]
        Uz = x_coord[:, 0] * u[:, 2]
        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        Uz = Uz.reshape(Uz.shape[0], 1)
        u_pred = torch.cat((Ux, Uy, Uz), -1)
        return u_pred

    def loss_function(self, step, epoch, x_scaled, edge_index, x_coord, X1_idx,traction, integrationIE, integrationEE):
        U = self.getU(x_coord, edge_index)

        if Example == 'Hyperelastic':
            if INT_TYPE == 'AD':
                internal = psi(U, x_coord, integrationIE, dx, dy, dz, shape)
            else:
                internal = psi_Gauss(U, x_coord, integrationIE, dx, dy, dz, shape)
        elif Example == 'Elastic':
            if INT_TYPE == 'AD':
                internal = LE(U, x_coord, integrationIE, dx, dy, dz, shape)
            else:
                internal = LE_Gauss(U, x_coord, integrationIE, dx, dy, dz, shape)


        neu_uP_pred = U[X1_idx,:3]
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
base  = './' + Example + '_' + INT_TYPE + '/'
if not os.path.exists( base ):
    os.mkdir( base )


# ----------------------------- define structural parameters ----------------------------------------
Length = 4.0
Width  = 1.0
Depth  = 1.0
numb_nodes_cont_param = 10
Ny = numb_nodes_cont_param
Nz = numb_nodes_cont_param
Nx = int((numb_nodes_cont_param-1) * int(Length/Width) + 1)

# Nx = 44; Ny = 13; Nz = 13

shape = [Nx, Ny, Nz]
print( shape )

x_min, y_min, z_min = (0.0, 0.0, 0.0)
(dx, dy, dz) = (Length / (Nx - 1), Width / (Ny - 1), Depth / (Nz - 1))

# --------------------- Graph and Data ---------------------------
G, Boundaries = setup_domain()
print('# of nodes is ', G.number_of_nodes())
print('# of X1 surface nodes is ', len(Boundaries['X1']))
print('# of X2 surface nodes is ', len(Boundaries['X2']))
print('# of Y1 surface nodes is ', len(Boundaries['Y1']))
print('# of Y2 surface nodes is ', len(Boundaries['Y2']))
print('# of Z1 surface nodes is ', len(Boundaries['Z1']))
print('# of Z2 surface nodes is ', len(Boundaries['Z2']))

# create edge index
adj = nx.to_scipy_sparse_array(G).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row,col], dim=0)

dataset = MeshDataSet()
data = dataset[0]
nodes =  data.to(device)
X1_idx = Boundaries['X1'].to(device)
# ----------------------------- End ----------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------

# ----------------------- network settings -----------------
D_in  = 3
H     = 16
D_out = 3

# Material Parameters
YM =  1000
PR =  0.3

# Loading
total_traction = -25.
step_max   = 50
ref_file = './AbaqusReferenceDisplacements/' + 'NH_Disp25_'

# Training
epochs = 20
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

    gem = DeepMixedMethod([D_in, H, D_out,lr,act_fn])
    start_time = time.time()
    disp_history, strainCG_histroy, stressC_history, stressC_last, strain_last , electric_potential_last , E_field , LOSS = gem.train_model()
    end_time = time.time()
    print('simulation time = ' + str(end_time - start_time) + 's')

    #######################################################################################################################################
    # Save data
    x_space = np.expand_dims(nodes.nodes[:,0].detach().cpu().numpy(), axis=1)
    y_space = np.expand_dims(nodes.nodes[:,1].detach().cpu().numpy(), axis=1)
    z_space = np.expand_dims(nodes.nodes[:,2].detach().cpu().numpy(), axis=1)
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
    Nodal_E = Nodal_E.cpu().detach().numpy()
    stress_vMis  = stress_vMis.unsqueeze(1).cpu().detach().numpy()
    electric_potential = electric_potential_last.unsqueeze(1).cpu().detach().numpy()


    Data = np.concatenate((coordin, U, Nodal_Strain , Nodal_Stress, stress_vMis, electric_potential , Nodal_E ), axis=1)
    np.save( base + 'Results.npy',Data)

    LBFGS_loss_D1 = np.array(LOSS[1])
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
    names = [ 'Ux' , 'Uy' , 'Uz' , 'E11' , 'E22' , 'E33' , 'E12' , 'E23' , 'E13' , 'S11' , 'S22' , 'S33' , 'S12' , 'S23' , 'S13' , 'SvM' , 'E_pot' , 'D1' , 'D2' , 'D3' ]
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