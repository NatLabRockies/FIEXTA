# -*- coding: utf-8 -*-
"""
Test LiSBOA scan optimization
"""

from lisboa import scan_optimizer as opt
from matplotlib import pyplot as plt
import numpy as np
plt.close('all')

#%% Inputs

#user
case='2D-single'
parallel=True

#Pareto (common)
azi1=[50,70]
azi2=[130,110]
dazi=[1,2]

#pareto (2D)
coords_2d='xy'
ele1_2d=[0,0]
ele2_2d=[0,0]
dele_2d=[0,0]

config_2d={'sigma':0.25,
        'mins':[0,-1000],
        'maxs':[500,500],
        'Dn0':[100,25],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1,
        'max_iter':3}

#pareto (3D)
coords_3d='xyz'
ele1_3d=[0,0]
ele2_3d=[5,10]
dele_3d=[0.5,1]

config_3d={'sigma':0.25,
        'mins':[100,-500,0],
        'maxs':[1000,500,200],
        'Dn0':[100,100,25],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1,
        'max_iter':5}

#lidar settings
ppr=1000
dr=30
rmin=100
rmax=1000
path_config_lidar='C:/Users/sletizia/Software/FIEXTA/halo_suite/halo_suite/configs/config.217.yaml'
mode='CSM'

#time info
T=600#[s] scan duration
tau=5#[s] integral timescale

#%% Initalization
if case=='2D-multiple':
    azi1={'s1':azi1,'s2':np.array(azi1)-180}
    azi2={'s1':azi2,'s2':np.array(azi2)-180}
    ele1_2d={'s1':ele1_2d,'s2':ele1_2d}
    ele2_2d={'s1':ele2_2d,'s2':ele2_2d}
    dazi={'s1':dazi,'s2':dazi}
    dele_2d={'s1':dele_2d,'s2':dele_2d}
    x0={'s1':0,'s2':1000}
    y0={'s1':0,'s2':-50}
    z0={'s1':0,'s2':0}
    path_config_lidar={'s1':path_config_lidar,'s2':path_config_lidar}
if case=='3D-multiple':
    azi1={'s1':azi1,'s2':np.array(azi1)-180}
    azi2={'s1':azi2,'s2':np.array(azi2)-180}
    ele1_3d={'s1':ele1_3d,'s2':ele1_3d}
    ele2_3d={'s1':ele2_3d,'s2':ele2_3d}
    dazi={'s1':dazi,'s2':dazi}
    dele_3d={'s1':dele_3d,'s2':dele_3d}
    x0={'s1':0,'s2':1000}
    y0={'s1':0,'s2':-50}
    z0={'s1':0,'s2':0}
    path_config_lidar={'s1':path_config_lidar,'s2':path_config_lidar}

#%% Main
if case=='2D-single':
    scopt=opt.scan_optimizer(config_2d)
    Pareto=scopt.pareto(coords_2d,0,0,0,azi1, azi2, ele1_2d, ele2_2d, dazi, dele_2d,None,None, volumetric=False,
                        rmin=rmin,rmax=rmax, T=T,tau=tau,mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar,
                        parallel=parallel)
elif case=='3D-single':
    scopt=opt.scan_optimizer(config_3d)
    Pareto=scopt.pareto(coords_3d,0,0,0,azi1, azi2, ele1_3d, ele2_3d, dazi, dele_3d,None,None, volumetric=True,
                        rmin=rmin,rmax=rmax, T=T,tau=tau,mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar,
                        parallel=parallel)
elif case=='2D-multiple':
    scopt=opt.scan_optimizer(config_2d)
    Pareto=scopt.pareto(coords_2d,x0,y0,z0,azi1, azi2, ele1_2d, ele2_2d, dazi, dele_2d,None,None, volumetric=False,
                        rmin=rmin,rmax=rmax, T=T,tau=tau,mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar,
                        parallel=parallel)
elif case=='3D-multiple':
    scopt=opt.scan_optimizer(config_3d)
    Pareto=scopt.pareto(coords_3d,x0,y0,z0,azi1, azi2, ele1_3d, ele2_3d, dazi, dele_3d,None,None, volumetric=True,
                        rmin=rmin,rmax=rmax, T=T,tau=tau,mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar,
                        parallel=parallel)