'''
Build daily scan schedule
'''

import os
cd=os.path.dirname(__file__)
import datetime as dt
import math
import numpy as np
from tkinter import Tk
from halo_suite.utilities import scan_file_compiler
from tkinter.filedialog import askopenfilename
from halo_suite import halo_simulator as hls
import yaml
import pandas as pd

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source=askopenfilename(
    title="Scan schedule file",
    filetypes=[("All files", "*.xlsx")],
    initialdir=cd,
)
assert os.path.isfile(source), f'Invalid file "{source}"' 

path_config_lidar=askopenfilename(
    title="Lidar configuration file",
    filetypes=[("All files", "*.yaml")],
    initialdir=cd,
)
print(f'Invalid file "{path_config_lidar}"')

time_corr=np.float64(input('Buffer time [%]: '))

#%% Initialization

#read schedule specs
schedule=pd.read_excel(source) 
filenames=schedule['Scan file'].values
T=schedule['Total time [s]'].values

#read lidar config
if os.path.isfile(path_config_lidar):
    with open(path_config_lidar, 'r') as fid:
        config = yaml.safe_load(fid)  

#file system
save_path=os.path.join(os.path.dirname(source),os.path.basename(source)[:-5]+'.dss')
os.makedirs(os.path.dirname(save_path),exist_ok=True)

#zeroing
scan_files=[]
avg_loops=np.zeros(len(filenames),dtype=int)
tau_s=np.zeros(len(filenames))
modes= np.empty(len(filenames), dtype="U20")

#%% Main
ctr=0
for f in filenames:
    ppr=schedule['PPR'].values[ctr]
    mode=schedule['Mode'].values[ctr]
    
    
    if os.path.splitext(f)[-1]=='.txt':
        scan_file=f
        tau_s[ctr]=T[ctr]
        time_corr=0
        modes[ctr]=mode
        volumetric=False
        
        if mode=='SSM':
            avg_loops[ctr]=int(ppr)
        elif mode=='CSM':
            avg_loops[ctr]=int(ppr/1000)
             
    elif os.path.splitext(f)[-1]=='.xlsx':
        volumetric=schedule['Volumetric'].values[ctr]=='TRUE'
        
        #load geometry from file
        scan=pd.read_excel(os.path.join(os.path.dirname(source),f))
        azi=scan['Azimuth [deg]'].values
        ele=scan['Elevation [deg]'].values
       
        #compile single scan file
        if mode=='SSM':
            scan_file=scan_file_compiler(mode=mode.upper(),azi=azi,ele=ele,repeats=1,
                                         identifier=os.path.splitext(f)[0],save_path=os.path.dirname(save_path),
                                         volumetric=volumetric,reset=True)
        elif mode=='CSM':
            scan_file=scan_file_compiler(mode=mode.upper(),azi=azi,ele=ele,repeats=1,ppr=ppr,
                                identifier=os.path.splitext(f)[0],save_path=os.path.dirname(save_path),
                                config=config,
                                optimize=True,volumetric=volumetric,reset=True)
            scan_files.append(os.path.basename(scan_file))
        
        #simulate scanning head
        if mode=='SSM':
            avg_loops[ctr]=int(ppr)
            
            halo_sim=hls.halo_simulator(config={'processing_time':config['Dt_p_SSM'],
                                                 'acquisition_time':config['Dt_a_SSM'],
                                                 'dwell_time': 0})
            
            t2,azi2,ele2,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode='SSM',ppr=ppr,source=scan_file,
                                                                          S_azi=config['S_azi_SSM'],
                                                                          A_azi=config['A_azi_SSM'],
                                                                          S_ele=config['S_ele_SSM'],
                                                                          A_ele=config['A_ele_SSM'])
            
        elif mode=='CSM':
            avg_loops[ctr]=int(ppr/1000)#in CSM, avg loops are derived from PPR in scan file
            halo_sim=hls.halo_simulator(config={'processing_time':config['Dt_p_CSM'],
                                                'acquisition_time':config['Dt_a_CSM'],
                                                'dwell_time':config['Dt_d_CSM'][ppr],
                                                'ppd_azi':config['ppd_azi'],
                                                'ppd_ele':config['ppd_ele']})
            
            t2,azi2,ele2,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode='CSM',ppr=ppr,source=scan_file)
        
        #calculat reps
        tau_s[ctr]=t2[-1]
        reps=np.floor(T[ctr]/(tau_s[ctr]*(1+time_corr/100))).astype(int)
        modes[ctr]=mode
        
        #compile repeated scan file
        if mode=='SSM':
            scan_file=scan_file_compiler(mode=mode.upper(),azi=azi,ele=ele,repeats=reps,
                                         identifier=os.path.splitext(f)[0],save_path=os.path.dirname(save_path),
                                         volumetric=volumetric,reset=False)
        elif mode=='CSM':
            scan_file=scan_file_compiler(mode=mode.upper(),azi=azi,ele=ele,repeats=reps,ppr=ppr,
                                identifier=os.path.splitext(f)[0],save_path=os.path.dirname(save_path),
                                config=config,
                                optimize=True,volumetric=volumetric,reset=False)
    
    scan_files.append(os.path.basename(scan_file))
    ctr+=1
    
#add reps and total time
N=np.floor(T/(tau_s*(1+time_corr/100))).astype(int)
schedule['Reps']=N
schedule['Total time (real) [s]']=tau_s*N

#adjust average loops for SSM
ppr_common=int(math.gcd(*avg_loops[modes=='SSM']))
if ppr_common>0:
    avg_loops[modes=='SSM']=np.int64(avg_loops[modes=='SSM']/ppr_common)
    print(f'Run schedule with {ppr_common} PPR.')
     
#write dss file
N_day=np.floor(24*3600/np.sum(T)).astype(int)
s=''
for i_seq in range(N_day):
    for i_scan in range(len(T)):
        t=np.float64(i_seq*np.sum(T)+np.sum(T[:i_scan]))
        t_str=dt.datetime.strftime(dt.datetime(2000, 1, 1)+dt.timedelta(seconds=t),'%H%M%S')
        s+=t_str+'\t'+scan_files[i_scan][:-4]+'\t'+str(int(avg_loops[i_scan]))+'\t'+modes[i_scan][0].upper()+'\t7\n'
        
with open(save_path,'w') as fid:
    fid.write(s[:-1])
    
schedule.set_index('Scan name').to_excel(source)