# -*- coding: utf-8 -*-
"""
Lidar Statistical Barnes Objective Analysis
"""

from scipy.special import gamma
from typing import Union, Optional
from scipy.interpolate import interpn
import itertools
import numpy as np
from lisboa.utilities import mid, get_logger, with_logging, _load_configuration
from lisboa.config import LisboaConfig

class statistics:
    def __init__(self,
                 config: Union[str, dict, LisboaConfig],
                 verbose: bool = True,
                 logger: Optional[object] = None,
                 logfile: Optional[str] = None):
        
        self.logger = get_logger(verbose=verbose, logger=logger,filename=logfile)
        self.logger.log("Initializing LiSBOA statistics")
        self.verbose=verbose
    
        # Load configuration based on input type
        self.config,exit_flag = _load_configuration(config)
        self.logger.log(exit_flag)
        if self.config is None:
            return
        else:
            LisboaConfig.validate(self.config)
    
    @with_logging    
    def calculate_weights(
            self,
            x_exp: list,
            f=None):
        
        #reject non valid points
        n=len(self.config.Dn0) 
        if f is None:
            real=~np.isnan(np.sum(np.array(x_exp),axis=0))
        else:
            real=~np.isnan(np.sum(np.array(x_exp),axis=0)+f)
            f=f[real]
            
        for j in range(n):
            x_exp[j]=x_exp[j][real]

        #initialize variables
        Dn0=np.array(self.config.Dn0)   
        N=len(x_exp[0])
        x=np.zeros((n,N))
        xc=np.zeros(n)
        X_edg=[];
        X_cen=[];
        grid=[]
        
        self.logger.log(f"Calculating LiSBOA weights for {N} experimental points")
      
        #LiSBOA setup
        dx=self.config.grid_factor*Dn0
        r_max=self.config.r_max*self.config.sigma
        V=np.pi**(n/2)/gamma(n/2+1)*r_max**n
        dist_edge=int(self.config.dist_edge*self.config.sigma/self.config.grid_factor)
        
        #grid definition
        for j in range(n):
            xc[j]=np.mean(x_exp[j])#centroid
            x[j]=(x_exp[j]-xc[j])/Dn0[j]#normalized coordinate  
            X_edg.append((np.arange(self.config.mins[j]-dx[j]/2,self.config.maxs[j]+dx[j]*1.01,dx[j])-xc[j])/Dn0[j])#cell edges
            X_cen.append(mid(X_edg[j]))#cell center

        X=np.meshgrid(*[X_cen[j] for j in range(n)], indexing='ij')#broadcasting
        
        #grid in dimensional form
        for j in range(n):
            grid.append(X_cen[j]*Dn0[j]+xc[j])
            
        #zeroing
        w=np.zeros(np.shape(X[0]),dtype=object)
        sel=np.zeros(np.shape(X[0]),dtype=object)
        Dd=np.zeros(np.shape(X[0]))
        edge=Dd==1
        nodes=np.where(X[0])
        ctr=0
        
        #loop over all grid points
        for i in zip(*[xx for xx in nodes]):
            #squared Euclidean distance from obs points
            distSq=0
            for j in range(n):
                distSq+=(x[j]-X[j][i])**2
            
            #weights
            s=np.where(distSq<r_max**2)[0]   
            if len(s)>0:
                w[i]=np.exp(-distSq[s]/(2*self.config.sigma**2))
            
            #local spacing
            if Dd[i]!=10**99:   
                if len(s)>1: 
                    
                    #collapse points closer than tol_dist
                    pos_uni=np.round(x[:,s]/self.config.tol_dist)*self.config.tol_dist
                    N_uni= len(np.unique(pos_uni,axis=1)[0,:])
                    
                    #calculate local spacing assuming isotropy
                    if N_uni>1:
                        Dd[i]=V**(1/n)/(N_uni**(1/n)-1)
                    else:
                        Dd[i]=10**99
                else:
                    Dd[i]=10**99
                    
            #store
            sel[i]=s
            
            ctr+=1
            if self.verbose:
                if int(ctr/1000)==ctr/1000:
                    self.logger.log(f"{ctr}/{len(X[0].ravel())} grid points done")
                    
        #find edge points and set spacing to infinity
        for i in zip(*[xx for xx in nodes]):
            ind_inf=[]
            if Dd[i]>self.config.max_Dd:
                for j in range(n):
                    i1=max(i[j]-dist_edge,0)
                    i2=min(i[j]+dist_edge,np.shape(X[0])[j]-1)
                    ind_inf.append(np.arange(i1,i2+1).astype(int))                
                for i_inf in itertools.product(*[ii for ii in ind_inf]):
                    edge[i_inf]=True
        Dd[edge]=10**99
        
        #undersampled region
        excl=Dd>self.config.max_Dd

        return grid,Dd,excl,w,sel,x_exp,f
    
    @with_logging
    def calculate_statistics(self,
                            x_exp: list,
                            f=None,
                            order=2):
        
        grid,Dd,excl,w,sel,x_exp,f=self.calculate_weights(x_exp,f)
        
        #zeroing
        df=f
        nodes=np.where(Dd)

        #iterations
        for m in range(self.config.max_iter):
            self.logger.log(f'Calculating statistics: iteration {m+1}/{self.config.max_iter}')
            WM=np.zeros(np.shape(Dd))+np.nan
            for i in zip(*[xx for xx in nodes]):
                if not excl[i]:
                    fs=np.array(df[sel[i]])
                    ws=np.array(w[i])
                    reals=~np.isnan(fs+ws)
                    if np.sum(reals)>0:
                        fs=fs[reals]
                        ws=ws[reals]                     
                        WM[i]=np.sum(fs*ws)/np.sum(ws)                       
            if m==0:
                avg=WM.copy()
            else:
                avg+=WM
            
            #residual
            df=f-interpn(tuple(grid),avg,np.array(x_exp).T,bounds_error=False,fill_value=np.nan)
            
        #HOM
        hom=np.zeros(np.shape(Dd))+np.nan
        for i in zip(*[xx for xx in nodes]):
            if not excl[i]:
                fs=np.array(df[sel[i]])
                ws=np.array(w[i])
                reals=~np.isnan(fs+ws)
                if np.sum(reals)>0:
                    fs=fs[reals]
                    ws=ws[reals]                     
                    hom[i]=np.sum(fs**order*ws)/np.sum(ws)      
                
        return grid,Dd,excl,avg,hom
   