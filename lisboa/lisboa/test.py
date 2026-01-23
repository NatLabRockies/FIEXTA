# -*- coding: utf-8 -*-
"""
Test LiSBOA
"""
import numpy as np
from lisboa import statistics as stats
from matplotlib import pyplot as plt
plt.close('all')

#%% Inputs
azi=np.arange(-25,26,5)+90
ele=azi*0
r=np.arange(0,1000,30)+15

config={'sigma':0.25,
        'mins':[0,-200],
        'maxs':[1000,200],
        'Dn0':[200,50],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1,
        'max_iter':3}

#%% Initialization
x=np.outer(np.cos(np.radians(ele))*np.cos(np.radians(90-azi)),r)
y=np.outer(np.cos(np.radians(ele))*np.sin(np.radians(90-azi)),r)
f=(x/10)+(y/100)**2
f=(f-np.min(f))/np.ptp(f)

x_exp=[x.ravel(),y.ravel()]

#%% Main
lproc=stats.statistics(config)
grid,Dd,excl,avg,hom=lproc.calculate_statistics(x_exp,f.ravel())
avg[excl==1]=np.nan

#%% Plots
plt.figure()
X,Y=np.meshgrid(grid[0],grid[1],indexing='ij')
plt.pcolor(X,Y,avg,vmin=0,vmax=1,edgecolors='k')
plt.scatter(x,y,c=f,s=10,vmin=0,vmax=1,edgecolor='k')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.grid()






