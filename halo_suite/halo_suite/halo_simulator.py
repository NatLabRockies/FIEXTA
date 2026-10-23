# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 14:21:06 2025

@author: sletizia
"""
import numpy as np
import os
import re
class halo_simulator:
    '''
    Simulator of the movement of the lidar scanning head
    '''
    def __init__(self,
                 config: dict,
                 source_scan: str=''):
    
        self.config=config
    
    def scanning_head_sim(self,
                          mode: str='SSM',
                          ppr: int=1000,
                          source: str='',
                          azi: np.ndarray = np.array([]),
                          ele: np.ndarray = np.array([]),
                          S_azi: float=36,
                          S_ele: float=72,
                          A_azi: float=36,
                          A_ele: float=72,
                          dt: float=0.01,
                          ang_tol: float=0.1):
        '''
        This module takes as input either:
            1) for SSM either a txt scan file and a sequence of azimuths and elevations and a single value of speeds and accelrations
            2) for CSM, either a txt scan file or a sequence of azimuths and elevations stops, and a sequence of speeds and accelerations
        '''
        #extract config
        Dt_p=self.config['processing_time']
        Dt_a=self.config['acquisition_time']
        Dt_d=self.config['dwell_time']
        Dt_s=Dt_p+Dt_a*ppr
        
        #read scan file
        if mode=='SSM':
            if os.path.isfile(source):
                with open(source,'r') as fid:
                    lines = fid.readlines()
                lines = [line.strip() for line in lines]
                #extract geometry
                azi=[]
                ele=[]
                for l in lines:
                    azi=np.append(azi,np.float32(l[:7]))
                    ele=np.append(ele,np.float32(l[7:]))
            
        elif mode=='CSM':
            ppd1=self.config['ppd_azi']
            ppd2=self.config['ppd_ele']
            
            if os.path.isfile(source):
                with open(source,'r') as fid:
                    lines = fid.readlines()
                lines = [line.strip() for line in lines]
                #extract kinematic parameters
                P1=[]
                P2=[]
                S1=[]
                S2=[]
                A1=[]
                A2=[]
                for l in lines:
                    if len(l)>10:
                        kin = [np.float64(v[1:]) for v in re.findall(r'=[-]?\d*\.?\d+', l)]
                    
                        A1=np.append(A1,kin[0])
                        S1=np.append(S1,kin[1])
                        P1=np.append(P1,kin[2])
                        A2=np.append(A2,kin[3])
                        S2=np.append(S2,kin[4])
                        P2=np.append(P2,kin[5])
                        
                azi=-np.round(P1/ppd1/ang_tol)*ang_tol
                ele=-np.round(P2/ppd2/ang_tol)*ang_tol
                S_azi=S1[1:]*10/ppd1
                S_ele=S2[1:]*10/ppd2
                A_azi=A1[1:]*1000/ppd1
                A_ele=A2[1:]*1000/ppd2
        
        #zeroing
        t_start=0
        t_all=[]
        azi_all=[]
        ele_all=[]
       
        if mode=='SSM':
            t_sim=[0]
        elif mode=='CSM':
            t_sim=[]
            
        azi_sim=[]
        ele_sim=[]
        
        S_azi+=np.zeros((len(azi))-1)
        S_ele+=np.zeros((len(azi))-1)
        A_azi+=np.zeros((len(azi))-1)
        A_ele+=np.zeros((len(azi))-1)
        
        #loop through segments
        for azi1,azi2,ele1,ele2,S1,S2,A1,A2 in zip(azi[:-1],azi[1:],ele[:-1],ele[1:],S_azi,S_ele,A_azi,A_ele):
            dazi=azi2-azi1
            dele=ele2-ele1
            if np.abs(dazi)>=ang_tol and np.abs(dele)<ang_tol:#PPI simulation
                _t,_azi=self.step_scanning_head(azi1,azi2,S1,A1,mode=mode)
                _ele=ele1+_azi*0
            elif np.abs(dazi)<ang_tol and np.abs(dele)>=ang_tol:#RHI simulations
                _t,_ele=self.step_scanning_head(ele1,ele2,S2,A2,mode=mode)
                _azi=azi1+_ele*0
            elif np.abs(dazi)>=ang_tol and np.abs(dele)>=ang_tol: #two-axis motion simulations
                _t_azi,_azi=self.step_scanning_head(azi1,azi2,S1,A1,mode=mode)
                _t_ele,_ele=self.step_scanning_head(ele1,ele2,S2,A2,mode=mode)
                if _t_azi[-1]>=_t_ele[-1]:
                    _t=_t_azi
                    _ele=np.append(_ele,np.zeros(np.sum(_t_azi>_t_ele[-1]))+_ele[-1])
                elif _t_azi[-1]<_t_ele[-1]:
                    _t=_t_ele
                    _azi=np.append(_azi,np.zeros(np.sum(_t_ele>_t_azi[-1]))+_azi[-1])
            elif np.abs(dazi)<ang_tol and np.abs(dele)<ang_tol:#stare
                    _t=[0,0]
                    _azi=[azi1,azi2]
                    _ele=[ele1,ele2]
                
            
            
            if mode=='SSM':
                #in SSM, add acquisition delay
                t_all=np.append(t_all,t_start+_t+Dt_s)
                t_sim=np.append(t_sim,t_start+_t[-1]+Dt_s)
                azi_all=np.append(azi_all,_azi)
                ele_all=np.append(ele_all,_ele)
                
            elif mode=='CSM':
                
                #dead time
                _t=np.append(_t,_t[-1]+Dt_d)
                _azi=np.append(_azi,_azi[-1])
                _ele=np.append(_ele,_ele[-1])
                
                #stacking
                t_int=np.arange(t_start,t_start+_t[-1]+Dt_s,Dt_s)
                t_sim=np.append(t_sim,t_int)
                t_all=np.append(t_all,t_start+_t)
                azi_all=np.append(azi_all,_azi)
                ele_all=np.append(ele_all,_ele)
                
                #interpolation at sampling location
                c=np.interp(t_int,t_all,np.cos(np.radians(azi_all)))
                s=np.interp(t_int,t_all,np.sin(np.radians(azi_all)))
                azi_sim=np.append(azi_sim,np.degrees(np.arctan2(s,c)))
                c=np.interp(t_int,t_all,np.cos(np.radians(ele_all)))
                s=np.interp(t_int,t_all,np.sin(np.radians(ele_all)))
                ele_sim=np.append(ele_sim,np.degrees(np.arctan2(s,c)))
            
            t_start=t_all[-1]
                
        azi_all=azi_all[1:]%360
        ele_all=ele_all[1:]
        t_all=t_all[1:]-t_all[1]
        
        if mode=='SSM':
            azi_sim=azi.copy()
            ele_sim=ele.copy()
            
        azi_sim=azi_sim%360
            
        return t_sim,azi_sim,ele_sim,t_all,azi_all,ele_all
    
    def step_scanning_head(self,ang1,ang2,S,A,mode='SSM',dt=0.01):
        
        #fix 0 values
        if A==0:
            A+=10**-10
        if S==0:
            S+=10**-10
        
        #if SSM mode is on and path crosses 0 degrees, find shortest route
        if mode=='SSM':
            if ang2-ang1<-180:
                ang1-=360
            elif ang2-ang1>180:
                   ang2-=360     
        
        sign=np.sign(ang2-ang1)
        
        if np.abs(ang2-ang1)<S**2/A:#for angular ranges shorter than the critical value
            Dt_m=2*(np.abs(ang2-ang1)/A)**0.5
            t=np.arange(0,Dt_m+dt,dt)
            ang=np.zeros(len(t))
            t1=Dt_m/2
            ang[t<t1]=ang1+0.5*A*(t[t<t1])**2*sign
            ang[t>=t1]=ang1+(ang2-ang1)/2+A*t1*(t[t>=t1]-t1)*sign-0.5*A*(t[t>=t1]-t1)**2*sign
        else:#for angular ranges larger than the critical value
            Dt_m=np.abs(ang2-ang1)/S+S/A
            t=np.arange(0,Dt_m+dt,dt)
            ang=np.zeros(len(t))
            t1=S/A
            t2=Dt_m-S/A
            ang[t<t1]=ang1+0.5*A*(t[t<t1])**2*sign
            ang[(t>=t1)*(t<t2)]=ang1+S**2/(2*A)*sign+S*(t[(t>=t1)*(t<t2)]-t1)*sign
            ang[t>=t2]=         ang2-S**2/(2*A)*sign+S*(t[t>=t2]-t2)*sign-0.5*A*(t[t>=t2]-t2)**2*sign
            
        return t,ang
        
        
        