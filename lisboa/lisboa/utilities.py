# -*- coding: utf-8 -*-
"""
LiSBOA utilities
"""
from typing import Optional, Union
from lisboa.logger import SingletonLogger
import numpy as np
from functools import wraps
from lisboa.config import LisboaConfig
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import warnings
import xarray as xr
warnings.filterwarnings("ignore", category=SyntaxWarning)

def get_logger(
        name: str = None, verbose: bool = True, logger: Optional[object] = None, filename=None
               ) -> SingletonLogger:
    """Utility function to get or create a logger instance."""
    
    #get logger only if it exists, otherwise create one
    if logger is None:
        logger = SingletonLogger(logger=logger, verbose=verbose,filename=filename)
    return logger


def with_logging(func):
    """Decorator to add logging to any class method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = self.logger

        if logger.verbose:
            class_name = self.__class__.__name__
            func_name = func.__name__
            logger.log(f"Calling {class_name}.{func_name}")

        try:
            result = func(self, *args, **kwargs)
            return result

        except Exception as e:
            if logger.verbose:
                class_name = self.__class__.__name__
                func_name = func.__name__
                logger.log(
                    f"Error in {class_name}.{func_name}: {str(e)}", level="error"
                )
            raise

    return wrapper

def _load_configuration(config: Union[dict, LisboaConfig]):
    """
    Load configuration from either dictionary, or LisboaConfig object.

    Args:
        config (dict, or LisboaConfig): Configuration source

    Returns:
        LisboaConfig or None: Configuration parameters or None if loading fails
    """
    try:
        if isinstance(config, LisboaConfig):
            return config, "Configuration successfully loaded"
        elif isinstance(config, dict):
            return LisboaConfig(**config), "Configuration successfully loaded"
        else:
            return None, f"Invalid config type. Expected dict or LisboaConfig, got {type(config)}"
            
    except Exception as e:
        return None, f"Error loading configuration: {str(e)}"


def mid(x):
    '''
    Midpoint of 1-D vector
    '''
    return (x[1:]+x[:-1])/2

def cosd(x):
    '''
    Cosine in degrees
    '''
    return np.cos(x/180*np.pi)

def sind(x):
    '''
    Sine in degrees
    '''
    return np.sin(x/180*np.pi)


def sphere2cart(r,azi,ele):
    x=np.outer(r,np.cos(np.radians(ele))*np.cos(np.radians(90-azi)))
    y=np.outer(r,np.cos(np.radians(ele))*np.sin(np.radians(90-azi)))
    z=np.outer(r,np.sin(np.radians(ele)))
    return x,y,z


def visualize_scan(file,sites):
    '''
    Visualize scan with well-sampled regions.
    '''
            
    fig=plt.figure(figsize=(18,10))
    if len(sites)>1:
        gs = GridSpec(nrows=len(sites)+1, ncols=2, width_ratios=[1, 2], figure=fig)
    else:
        gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 2], figure=fig)

    colors=[]
    color_cycle=['k','r','m','c','o','brown','navy']
    for i in range(len(sites)):
        colors.append(color_cycle[int(np.mod(i,len(sites)))])
        
    #specific data
    info={}
    points={}
    
    if len(sites)>1:
        for s in sites:
            Data=xr.open_dataset(file,group=s,engine='netcdf4')
            
            p={}
            for c in Data.excl.coords:
                p[c]=Data[c+'_points']
            points[s]=p
            
            azi=Data.azi.values
            ele=Data.ele.values
            r=Data.r.values
            info[s]=r'$\alpha='+str(np.round(azi[0],2))+':'+str(np.round(np.diff(azi)[0],2))+':'+str(np.round(azi[-1],2))+r'^\circ$'+ '\n'+\
                 r'$\beta=' +str(np.round(ele[0],2))+':'+str(np.round(np.diff(ele)[0],2))+':'+str(np.round(ele[-1],2))+r'^\circ$' + '\n'+\
                 r'$r= '    +str(np.round(r[0],2))  +':'+str(np.round(np.diff(r)[0],2))  +':'+str(np.round(r[-1],2))+ '$ m' +'\n'+\
                 r'$\tau_s='+str(np.round(Data.attrs['duration'],1))+r'$ s'+ '\n'+\
                 'Scan mode = '+Data.attrs['mode']
     
        #common data
        Data=xr.open_dataset(file,group='synthesis',engine='netcdf4')
        info['synthesis']= r'$\epsilon_I='+str(np.round(Data.attrs['epsilon1'],2))+r'$'                      + '\n'+\
                           r'$\epsilon_{II}='+str(np.round(Data.attrs['epsilon2'],2))+r'$'                   + '\n'+\
                           r'$\Delta n_0=['+ ", ".join(f"{v:.2f}" for v in Data.attrs['config_Dn0']) + "]"+r'$ m'                                        
    else:
        Data=xr.open_dataset(file,engine='netcdf4')
        p={}
        for c in Data.excl.coords:
            p[c]=Data[c+'_points']
        points['']=p
        
        azi=Data.azi.values
        ele=Data.ele.values
        r=Data.r.values
        info['']=r'$\alpha='+str(np.round(azi[0],2))+':'+str(np.round(np.diff(azi)[0],2))+':'+str(np.round(azi[-1],2))+r'^\circ$'+ '\n'+\
             r'$\beta=' +str(np.round(ele[0],2))+':'+str(np.round(np.diff(ele)[0],2))+':'+str(np.round(ele[-1],2))+r'^\circ$' + '\n'+\
             r'$r= '    +str(np.round(r[0],2))  +':'+str(np.round(np.diff(r)[0],2))  +':'+str(np.round(r[-1],2))+ '$ m' +'\n'+\
             r'$\Delta n_0=['+ ", ".join(f"{v:.2f}" for v in Data.attrs['config_Dn0']) + "]"+r'$ m'                                         + '\n'+\
             r'$\epsilon_I='+str(np.round(Data.attrs['epsilon1'],2))+r'$'                      + '\n'+\
             r'$\epsilon_{II}='+str(np.round(Data.attrs['epsilon2'],2))+r'$'                   + '\n'+\
             r'$\tau_s='+str(np.round(Data.attrs['duration'],1))+r'$ s'+ '\n'+\
             'Scan mode = '+Data.attrs['mode']
             
    coords=''
    for c in Data.excl.coords:
        coords+=c
    
    excl=Data.excl
        
    ctr=0
    for s in sites:
        ax_text = fig.add_subplot(gs[ctr, 0])
        ax_text.axis("off")
        ax_text.text(0,0.9,s=info[s],color=colors[ctr],bbox={'edgecolor':'k','facecolor':(1,1,1,0.5)})
        ctr+=1
        
    if len(sites)>1:
        ax_text = fig.add_subplot(gs[-1, 0])
        ax_text.axis("off")
        ax_text.text(0,0.9,s=info['synthesis'],color='w',bbox={'edgecolor':'k','facecolor':'k'})
        
    #plot 2D scan
    if coords!='xyz':
        ax_geom = fig.add_subplot(gs[:, 1])
        
        fill=np.zeros(np.shape(excl))
        fill[excl==False]=10
        cmap_g = LinearSegmentedColormap.from_list("white_to_g", ["white", (0, 0.5, 0,0.1)])
        ax_geom.pcolor(Data[coords[0]],Data[coords[1]],fill.T,cmap=cmap_g,alpha=0.5)
        
        ctr=0
        for s in sites:
            ax_geom.plot(points[s][coords[0]],points[s][coords[1]],'.',color=colors[ctr],markersize=2,label=s)
            ctr+=1
            
        ax_geom.set_aspect('equal')
        ax_geom.set_xlim([Data.attrs['config_mins'][0],Data.attrs['config_maxs'][0]])
        ax_geom.set_ylim([Data.attrs['config_mins'][1],Data.attrs['config_maxs'][1]])
        ax_geom.set_xlabel(r'$'+str(coords[0])+'$ [m]')
        ax_geom.set_ylabel(r'$'+str(coords[1])+'$ [m]')
        dtick=np.min([np.diff(ax_geom.get_xticks())[0],
                      np.diff(ax_geom.get_yticks())[0]])
        ax_geom.set_xticks(np.arange(Data.attrs['config_mins'][0],Data.attrs['config_maxs'][0]+dtick,dtick))
        ax_geom.set_yticks(np.arange(Data.attrs['config_mins'][1],Data.attrs['config_maxs'][1]+dtick,dtick))
        ax_geom.grid(True)
        
    #plot 3D scan
    elif coords=='xyz':
        ax_geom = fig.add_subplot(gs[:, 1],projection='3d')
        
        x=Data[coords[0]].values
        y=Data[coords[1]].values
        z=Data[coords[2]].values
        
        dx=np.diff(x)[0]
        dy=np.diff(y)[0]
        dz=np.diff(x)[0]
        
        fill=excl.values==False
        X,Y,Z=np.meshgrid(np.append(x-dx/2,x[-1]+dx),np.append(y-dy/2,y[-1]+dy),np.append(z-dz/2,z[-1]+dz),indexing='ij')
        ax_geom.voxels(X,Y,Z,fill,facecolor='g',alpha=0.1)
        
        ctr=0
        for s in sites:
            x_exp=points[s]['x']
            y_exp=points[s]['y']
            z_exp=points[s]['z']
            sel_x=(x_exp>Data.attrs['config_mins'][0])*(x_exp<Data.attrs['config_maxs'][0])
            sel_y=(y_exp>Data.attrs['config_mins'][1])*(y_exp<Data.attrs['config_maxs'][1])
            sel_z=(z_exp>Data.attrs['config_mins'][2])*(z_exp<Data.attrs['config_maxs'][2])
            sel=sel_x*sel_y*sel_z
            ax_geom.plot(x_exp[sel],y_exp[sel],z_exp[sel],'.',markersize=2,color=colors[ctr],label=s)
            ctr+=1
            
        ax_geom.set_aspect('equal')
        ax_geom.set_xlim([Data.attrs['config_mins'][0],Data.attrs['config_maxs'][0]])
        ax_geom.set_ylim([Data.attrs['config_mins'][1],Data.attrs['config_maxs'][1]])
        ax_geom.set_zlim([Data.attrs['config_mins'][2],Data.attrs['config_maxs'][2]])
        ax_geom.set_xlabel(r'$x$ [m]',labelpad=10)
        ax_geom.set_ylabel(r'$y$ [m]',labelpad=10)
        ax_geom.set_zlabel(r'$z$ [m]',labelpad=10)
        dtick=np.max([np.diff(ax_geom.get_xticks())[0],
                      np.diff(ax_geom.get_yticks())[0],
                      np.diff(ax_geom.get_zticks())[0]])
        ax_geom.set_xticks(np.arange(Data.attrs['config_mins'][0],Data.attrs['config_maxs'][0]+dtick,dtick))
        ax_geom.set_yticks(np.arange(Data.attrs['config_mins'][1],Data.attrs['config_maxs'][1]+dtick,dtick))
        ax_geom.set_zticks(np.arange(Data.attrs['config_mins'][2],Data.attrs['config_maxs'][2]+dtick,dtick))
        
    plt.tight_layout()  
    if len(sites)>1:
        plt.legend(draggable=True)
        
    return fig

def visualize_Pareto(Data):
    
    #extract information
    num_ang=len(Data.index_ang)
    num_dang=len(Data.index_dang)
    epsilon1=Data.epsilon1.values
    epsilon2=Data.epsilon2.values
    
    #graphics
    cmap = plt.cm.jet
    colors = [cmap(v) for v in np.linspace(0,1,num_dang)]
    num_row=int(np.floor(num_ang**0.5))
    num_col=int(np.ceil(num_ang/num_row))
    fig=plt.figure(figsize=(18,8))
    
    #multiple Doppler
    if 'site' in Data.coords:
        for i_ang in range(num_ang):
            plt.subplot(num_row,num_col,i_ang+1)
            plt.plot(epsilon1[np.arange(num_ang)!=i_ang,:],epsilon2[np.arange(num_ang)!=i_ang,:],'.k',markersize=30,alpha=0.25)
            for i_dang in range(num_dang):
                label=''
                for s in Data.site:
                    if 'dazi' in Data:
                        dazi=Data.dazi.sel(site=s).values
                        dele=Data.dele.sel(site=s).values
                        label+=s.values+': '+r'$\Delta \alpha='+str(np.round(dazi[i_dang],2))+r'^\circ$, $\Delta \beta='+str(np.round(dele[i_dang],2))+r'^\circ$'+'\n'
                    elif 'num_azi' in Data:
                        num_azi=Data.num_azi.sel(site=s).values
                        num_ele=Data.num_ele.sel(site=s).values
                        label+=s.values+': '+r'$num\alpha='+str(np.round(num_azi[i_dang],2))+r'$, $num\beta='+str(np.round(num_ele[i_dang],2))+r'$'+'\n'
                plt.plot(epsilon1[i_ang,i_dang],epsilon2[i_ang,i_dang],'.',color=colors[i_dang],markeredgecolor='k',markersize=30,label=label[:-1])
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel(r'$\epsilon_I$')
            plt.ylabel(r'$\epsilon_{II}$')  
            title=''
            for s in Data.site:
                azi1=Data.azi1.sel(site=s).values
                azi2=Data.azi2.sel(site=s).values
                ele1=Data.ele1.sel(site=s).values
                ele2=Data.ele2.sel(site=s).values
                title+=s.values+': '+r'$\alpha \in ['+str(np.round(azi1[i_ang],2))+', '+str(np.round(azi2[i_ang],2))+r']^\circ$, $\beta \in ['+str(np.round(ele1[i_ang],2))+', '+str(np.round(ele2[i_ang],2))+r']^\circ$'+'\n'
            plt.title(title[:-1])
            plt.grid()
            
    #single Doppler  
    else:
        azi1=Data.azi1.values
        azi2=Data.azi2.values
        ele1=Data.ele1.values
        ele2=Data.ele2.values
        
        if 'dazi' in Data:
            dazi=Data.dazi.values
            dele=Data.dele.values
        elif 'num_azi' in Data:
            num_azi=Data.num_azi.values
            num_ele=Data.num_ele.values
            
        for i_ang in range(num_ang):
            plt.subplot(num_row,num_col,i_ang+1)
            plt.plot(epsilon1[np.arange(num_ang)!=i_ang,:],epsilon2[np.arange(num_ang)!=i_ang,:],'.k',markersize=30,alpha=0.25)
            for i_dang in range(num_dang):
                if 'dazi' in Data:
                    label=r'$\Delta \alpha='+str(np.round(dazi[i_dang],2))+r'^\circ$, $\Delta \beta='+str(np.round(dele[i_dang],2))+r'^\circ$'
                elif 'num_azi' in Data:
                    label=r'$num_\alpha='+str(np.round(num_azi[i_dang],2))+r'$, $num_\beta='+str(np.round(num_ele[i_dang],2))+r'$'
                plt.plot(epsilon1[i_ang,i_dang],epsilon2[i_ang,i_dang],'.',color=colors[i_dang],markeredgecolor='k',markersize=30,label=label)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel(r'$\epsilon_I$')
            plt.ylabel(r'$\epsilon_{II}$')  
            plt.title(r'$\alpha \in ['+str(np.round(azi1[i_ang],2))+', '+str(np.round(azi2[i_ang],2))+r']^\circ$, $\beta \in ['+str(np.round(ele1[i_ang],2))+', '+str(np.round(ele2[i_ang],2))+r']^\circ$')
            plt.grid()
    
    plt.tight_layout()
    plt.legend(draggable=True)
    
    return fig

def is_dict_of_dicts(d):
    return (
        isinstance(d, dict)
        and len(d) > 0
        and all(isinstance(v, dict) for v in d.values())
    )