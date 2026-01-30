# -*- coding: utf-8 -*-
"""
Scan optimizer using Pareto front based on LiSBOA results
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
from halo_suite import halo_simulator as hls 
from typing import Optional
from halo_suite.utilities import scan_file_compiler
from lisboa import statistics as stats
from lisboa.utilities import sphere2cart, visualize_Pareto, visualize_scan, get_logger, with_logging
from matplotlib import pyplot as plt
import xarray as xr
from datetime import datetime
from typing import Union
import yaml

class scan_optimizer:
    def __init__(self,
                 config: dict,
                 save_path: str=os.path.join(cd,'data','Pareto'),
                 save_data: bool=True,
                 make_figures: bool=True,
                 save_figures: bool=True,
                 identifier: str='',
                 verbose: bool = True,
                 logger: Optional[object] = None,
                 logfile: Optional[str] = None):
        
        self.logger = get_logger(verbose=verbose, logger=logger,filename=logfile)
        self.logger.log("Initializing LiSBOA scan optimizer")
        
        if identifier!='':
            identifier='.'+identifier
        self.config=config
        self.make_figures=make_figures
        self.save_figures=save_figures
        if save_data:
            self.save_name=os.path.join(save_path,datetime.strftime(datetime.now(),'%Y%m%d.%H%M%S')+identifier)
            os.makedirs(self.save_name)
     
    @with_logging    
    def pareto(
            self,
            coords,
            x0: Union[float,dict],
            y0: Union[float,dict],
            z0: Union[float,dict],
            azi1: Union[np.array,dict],
            azi2: Union[np.array,dict],
            ele1: Union[np.array,dict],
            ele2: Union[np.array,dict],
            dazi: Union[np.array,dict],
            dele: Union[np.array,dict],
            num_azi: Union[np.array,dict],
            num_ele: Union[np.array,dict],
            rmin: float,
            rmax: float,
            dr: float,
            ppr: int,
            volumetric: bool,
            mode: str,
            path_config_lidar: str,
            T: float,
            tau: float,
            full_scan_file: bool = False):
        
        #initialize LiSBOA
        lproc=stats.statistics(self.config,logger=self.logger)
        
        with open(path_config_lidar, 'r') as fid:
            config_lidar = yaml.safe_load(fid)  
            
        #select resolution mode
        if isinstance(dazi, np.ndarray) or isinstance(dazi, dict) \
       and isinstance(dele, np.ndarray) or isinstance(dele, dict):
            res_mode='degrees'
            res_azi=dazi
            res_ele=dele
            geom_info=[x0,y0,z0,azi2,azi2,ele1,ele2,dazi,dele]
                
        elif isinstance(num_azi, np.ndarray) or isinstance(num_azi, dict) \
         and isinstance(num_ele, np.ndarray) or isinstance(num_ele, dict):
            res_mode='count'
            res_azi=num_azi
            res_ele=num_ele
            geom_info=[x0,y0,z0,azi2,azi2,ele1,ele2,num_azi,num_ele]
        else:
            raise BaseException('Could not figure out angular resolution type (degrees or count).')
        
        #check single vs multiple Doppler
        if all(isinstance(x, dict) for x in geom_info):
            print(f'Optimizing geometry for multi-Doppler data from {len(azi1)} instruments.')
        elif all(isinstance(x, np.array) for x in geom_info):
            print('Optimizing geometry for single-Doppler data.')
            for x in geom_info:
                globals()[x] = {'ins1': globals()[x]}
        else:
            raise BaseException('All geometrical scan information have to be either np.array or dict.')
            
        #zeroing
        epsilon1=np.zeros((len(azi1),len(res_azi)))+np.nan
        epsilon2=np.zeros((len(azi1),len(res_azi)))+np.nan
        duration=np.zeros((len(azi1),len(res_azi)))+np.nan
    
        #scan testing
        r=np.arange(rmin,rmax+dr/2,dr)
        
        
        sites=list(azi1.keys())
        i_ang=0
        for i_ang in range(len(azi1[sites[0]])):#loop through angular sectors
            i_dang=0
            for i_dang in range(len(res_azi[sites[0]])):
                Dd_all=[]
                excl_all=[]
                duration_all=[]
                for s in azi1.keys():
                    a1=azi1[s][i_ang]
                    a2=azi2[s][i_ang]
                    e1=ele1[s][i_ang]
                    e2=ele2[s][i_ang]
                    ra=res_azi[s][i_ang]
                    re=res_ele[s][i_ang]
                    
                    self.logger.log(f"Evaluating azi={a1}:{ra}:{a2}, ele={e1}:{re}:{e2}")
                    
                    #expand azimuth and elevation vectors
                    if res_mode=='degrees':
                        ra+=10**-10
                        re+=10**-10
                        azi=np.arange(a1,a2+ra/2,ra)
                        ele=np.arange(e1,e2+re/2,re)
                        scan_name=f'{s}_{a1:.2f}_{ra:.2f}_{a2:.2f}_{e1:.2f}_{re:.2f}_{e2:.2f}'
                    elif res_mode=='count':
                        azi=np.linspace(a1,a2,ra)
                        ele=np.linspace(e1,e2,re)
                        scan_name=f'{s}_{a1:.2f}_{ra}_{a2:.2f}_{e1:.2f}_{re}_{e2:.2f}'
                        
                    if len(azi)==1:
                        azi=azi.squeeze()+np.zeros(len(ele))
                    if len(ele)==1:
                        ele=ele.squeeze()+np.zeros(len(azi))
                    
                    if mode=='SSM':
                        scan_file=scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,
                                                     identifier=scan_name,save_path=self.save_name,
                                                     volumetric=volumetric,reset=True)
                        
                        halo_sim=hls.halo_simulator(config={'processing_time':  config_lidar['Dt_p_SSM'],
                                                             'acquisition_time':config_lidar['Dt_a_SSM'],
                                                             'dwell_time': 0})
                        
                        t,azi_sim,ele_sim,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode=mode,ppr=ppr,
                                                                                      S_azi=config_lidar['S_azi_SSM'],
                                                                                      A_azi=config_lidar['A_azi_SSM'],
                                                                                      S_ele=config_lidar['S_ele_SSM'],
                                                                                      A_ele=config_lidar['A_ele_SSM'],
                                                                                      source=scan_file)
                    elif mode=='CSM':
                        scan_file=scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,ppr=ppr,
                                           identifier=scan_name,config=config_lidar,save_path=self.save_name,
                                           optimize=True,volumetric=volumetric,reset=True)
           
                        halo_sim=hls.halo_simulator(config={'processing_time': config_lidar['Dt_p_CSM'],
                                                            'acquisition_time':config_lidar['Dt_a_CSM'],
                                                            'dwell_time':      config_lidar['Dt_d_CSM'][ppr],
                                                            'ppd_azi':         config_lidar['ppd_azi'],
                                                            'ppd_ele':         config_lidar['ppd_ele']})
                    
                        t,azi_sim,ele_sim,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode=mode,ppr=ppr,source=scan_file)
                        
                        duration_all=np.append(duration_all,t[-1])
                        
                    x,y,z=sphere2cart(r, azi_sim, ele_sim)
                    if coords=='xy':
                        x_exp=[x.ravel(),y.ravel()]
                    elif coords=='xz':
                        x_exp=[x.ravel(),z.ravel()]
                    elif coords=='yz':
                        x_exp=[y.ravel(),z.ravel()]
                    elif coords=='xyz':
                        x_exp=[x.ravel(),y.ravel(),z.ravel()]

                    _,Dd,excl,grid,_,_,_=lproc.calculate_weights(x_exp)
                    Dd_all+=[Dd]
                    excl_all+=[excl]
                    
                #epsilon1
                Dd=np.max(np.stack(Dd_all,axis=3),axis=3)
                excl=np.min(np.stack(excl_all,axis=3),axis=3)
                epsilon1[i_ang,i_dang]=np.sum(excl)/np.size(excl)
                
                #epsilon2
                duration[i_ang,i_dang]=np.max(duration_all)
                if T>duration[i_ang,i_dang]:
                    L=int(np.floor(T/duration[i_ang,i_dang]))
                    p=np.arange(1,L)
                    epsilon2[i_ang,i_dang]=(1/L+2/L**2*np.sum((L-p)*np.exp(-t[-1]/tau*p)))**0.5
                    
                    if full_scan_file:
                        if mode=='SSM':
                           scan_file=scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=L,
                                                        identifier=f'{scan_name}',save_path=self.save_name,
                                                        volumetric=volumetric,reset=True)
                        elif mode=='CSM':
                            scan_file=scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=L,ppr=ppr,
                                               identifier=f'{scan_name}',config=config_lidar,save_path=self.save_name,
                                               optimize=True,volumetric=volumetric,reset=True)
                else:
                    epsilon2[i_ang,i_dang]=np.nan
          
                
                #save data
                if hasattr(self,'save_name'):
                    
                    space_coords={}
                    ctr=0
                    for c in coords:
                        space_coords[c]=grid[ctr]
                        ctr+=1
                        
                    Output=xr.Dataset()
                    Output['Dd']=xr.DataArray(Dd,coords=space_coords,
                                        attrs={'description':'Local normalized data spacing'})
                    Output['excl']=xr.DataArray(excl,coords=space_coords,
                                        attrs={'description':'Grid points failing the Peterson-Middleton test'})
                    Output['x_points']=xr.DataArray(x.ravel(),coords={'index':np.arange(len(x.ravel()))},
                                        attrs={'description':'x of experimental points'})
                    Output['y_points']=xr.DataArray(y.ravel(),coords={'index':np.arange(len(y.ravel()))},
                                        attrs={'description':'y of experimental points'})
                    Output['z_points']=xr.DataArray(z.ravel(),coords={'index':np.arange(len(z.ravel()))},
                                        attrs={'description':'z of experimental points'})
                    Output['azi']=xr.DataArray(azi,coords={'index_azi':np.arange(len(azi))},
                                               attrs={'description':'azimuth angle','units':'degrees'})
                    Output['ele']=xr.DataArray(ele,coords={'index_ele':np.arange(len(ele))},
                                               attrs={'description':'elevation angle','units':'degrees'})
                    Output['r']=xr.DataArray(r,coords={'index_r':np.arange(len(r))},
                                               attrs={'description':'range','units':'m'})
                    
                    Output.attrs={}
                    for c in self.config:
                        Output.attrs[f'config_{c}']=self.config[c]
                    Output.attrs['scan_file']=scan_file
                    
                    Output.attrs['epsilon1']=epsilon1[i_ang,i_dang]
                    Output.attrs['epsilon2']=epsilon2[i_ang,i_dang]
                    Output.attrs['duration']=duration[i_ang,i_dang]
                    Output.attrs['volumetric']=str(volumetric)
                    Output.attrs['mode']=mode
                    Output.to_netcdf(os.path.join(self.save_name,scan_name+'.nc'))
                    
                if self.make_figures:
                    fig=visualize_scan(Output)
                if self.save_figures:
                    fig.savefig(os.path.join(self.save_name,scan_name+'.png'))
                    plt.close(fig)
                    
                i_dang+=1
            i_ang+=1
            
        #build output
        Output=xr.Dataset()
        Output['epsilon1']=xr.DataArray(epsilon1,coords={'index_ang':np.arange(len(azi1)),'index_dang':np.arange(len(res_azi))},
                            attrs={'description':'fraction of undersampled volume'})
        Output['epsilon2']=xr.DataArray(epsilon2,coords={'index_ang':np.arange(len(azi1)),'index_dang':np.arange(len(res_azi))},
                            attrs={'description':'normalized error on the mean'})
        Output['duration']=xr.DataArray(duration,coords={'index_ang':np.arange(len(azi1)),'index_dang':np.arange(len(res_azi))},
                            attrs={'description':'Scan duration'})
        Output['azi1']=xr.DataArray(azi1,coords={'index_ang':np.arange(len(azi1))},
                            attrs={'description':'start azimuth','units':'degrees'})
        Output['azi2']=xr.DataArray(azi2,coords={'index_ang':np.arange(len(azi1))},
                            attrs={'description':'end azimuth','units':'degrees'})
        Output['ele1']=xr.DataArray(ele1,coords={'index_ang':np.arange(len(azi1))},
                            attrs={'description':'start elevation','units':'degrees'})
        Output['ele2']=xr.DataArray(ele2,coords={'index_ang':np.arange(len(azi1))},
                            attrs={'description':'end elevation','units':'degrees'})
        if res_mode=='degrees':
            Output['dazi']=xr.DataArray(res_azi,coords={'index_dang':np.arange(len(res_azi))},
                                attrs={'description':'azimuth step','units':'degrees'})
            Output['dele']=xr.DataArray(res_ele,coords={'index_dang':np.arange(len(res_azi))},
                                attrs={'description':'elevation step','units':'degrees'})
        elif res_mode=='count':
            Output['num_azi']=xr.DataArray(res_azi,coords={'index_dang':np.arange(len(res_azi))},
                                attrs={'description':'azimuth count','units':''})
            Output['num_ele']=xr.DataArray(res_ele,coords={'index_dang':np.arange(len(res_azi))},
                                attrs={'description':'elevation count','units':''})
        
        Output.to_netcdf(os.path.join(self.save_name,os.path.basename(self.save_name)+'.Pareto.nc'))
        
        if self.make_figures:
            fig=visualize_Pareto(Output)
        if self.save_figures:
            fig.savefig(os.path.join(self.save_name,os.path.basename(self.save_name)+'.Pareto.png'))
        
        self.logger.log(f'Pareto results saved as {os.path.basename(self.save_name)+".Pareto.nc"}')
            
        return Output
                     