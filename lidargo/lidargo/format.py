'''
Convert raw lidar files to netCDF in WDH-compatible netCDF format.

Part of this script is readapted from https://github.com/a2edap/ingest-awaken/tree/main/pipelines/lidar/lidar_halo_xrp under license LICENSE_BSD-2_Batelle.
'''
import os
import time
import xarray as xr
import numpy as np
import re
from typing import Union, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from lidargo.utilities import get_logger, with_logging, format_time_xticks,_load_configuration
from lidargo.config import LidarConfigFormat

class Format:
    def __init__(
        self,
        source: str,
        config: Union[str, dict, LidarConfigFormat],
        verbose: bool = True,
        logger: Optional[object] = None,
        logfile=None,
    ):
        """
        Initialize the LIDAR data processor with configuration parameters.

        Args:
            source (str): The input-level file name
            config (str, dict, or LidarConfig): Either a path to an Excel config file, a dictionary of configuration parameters, or a LidarConfig object
            verbose (bool, optional): Whether to print QC-related information. Defaults to True.
            logger (Logger, optional): Logger instance for logging messages. Defaults to None.
        """
        self.logger = get_logger(verbose=verbose, logger=logger,filename=logfile)
        self.source = source

        self.logger.log(
            f"Initializing formatting of {os.path.basename(self.source)}"
        )

        # Load configuration based on input type
        self.config,exit_flag = _load_configuration(config,self.source,'format')
        self.logger.log(exit_flag)
        if self.config is None:
            return
        else:
            LidarConfigFormat.validate(self.config)
    
    @with_logging
    def process_scan(self, save_file=True, save_path=None, replace=True,make_figures=True,save_figures=True):
        
        '''
        Format the raw scan file into WDH-compatible netCDF file
        
        Inputs:
        ------
        source: str
            The input-level file name 
        save_file: bool
            Whether or not save the final processed file
        save_path: str
            String with the directory the destination of the processed files.
            Optional, defaults to analogous input, replacing input-level with output-level
            data. Creates the necessary intermediate directories
        replace: bool
            Whether or not to replace processed scan if one already exists
        
        '''
        
        #rename raw file
        if self.config.model=='halo':
            source00=self.rename_halo_xr(self.source,self.config.site,self.config.z_id,save_path,replace)
        elif self.config.model=='windcube':
            source00=self.rename_windcube_200s(self.source,self.config.site,self.config.z_id,save_path,replace)
        else:
            self.logger.log(f"Lidar model {self.config.model} not supported")
            return
            
        #format renamed raw file
        save_filename = ('.'+self.config.data_level_out+'.').join(source00.split('.00.')).replace('.hpl','.nc')
        if save_path is not None:
            save_filename=os.path.join(save_path.replace('.00','.'+self.config.data_level_out),os.path.basename(save_filename))
        
        os.makedirs(os.path.dirname(save_filename),exist_ok=True)
        
        self.save_filename=save_filename
        
        if save_file and not replace and os.path.isfile(save_filename):
            self.logger.log(f'Processed file {save_filename} already exists, skipping it')
            return
        else:
            self.logger.log(f'Generating formatted file {os.path.basename(save_filename)}')
            
            if self.config.model=='halo':
                outputData=self.read_halo_xr(source00)
            elif self.config.model=='windcube':
                outputData=self.read_windcube_200s(source00)
            else:
                self.logger.log(f"Lidar model {self.config.model} not supported")
                return
                
            if save_file:
                outputData.to_netcdf(save_filename)
                self.logger.log(f'Formatted file saved as {save_filename}')
                
            if make_figures:
                self.plot_raw_data(outputData,save_figures)
        
    def _copy_raw_file(self,source,dest,stability_timeout=30,poll_interval=1):
        '''
        Copy source to dest, guarding against two failure modes seen when the
        source file is still being written by another process (e.g. an
        instrument data-transfer job) at the time of the copy:
        - the file grows mid-copy, so wait until its size is unchanged across
          two polls before starting
        - shutil.copyfile's Linux sendfile() fast path can report EOF (and so
          silently produce a truncated copy, with no exception raised) if the
          source shrinks/changes underneath it; verify sizes match afterward
        '''
        size=os.path.getsize(source)
        elapsed=0
        while elapsed<stability_timeout:
            time.sleep(poll_interval)
            elapsed+=poll_interval
            new_size=os.path.getsize(source)
            if new_size==size:
                break
            size=new_size
        else:
            self.logger.log(f"{source} did not stabilize within {stability_timeout} s, copying anyway",level="warning")

        shutil.copyfile(source,dest)

        dest_size=os.path.getsize(dest)
        if dest_size!=size:
            os.remove(dest)
            raise RuntimeError(f"Copy of {source} to {dest} is truncated ({dest_size} of {size} bytes); source may have been modified during copy")

    @with_logging
    def rename_halo_xr(self,source,site,z_id,save_path=None,replace=False):
        if save_path is None:
            save_path=os.path.dirname(source)
        os.makedirs(save_path,exist_ok=True)
        
        if 'Stare' in os.path.basename(source):
            pattern = r"Stare_\d+_(\d{8})_(\d{2})\.hpl"
            scan_type='stare'
        elif 'User' in os.path.basename(source):
            pattern = r"User\d{1}_\d+_(\d{8})_(\d{6})\.hpl"
            scan_type='user'+os.path.basename(source)[os.path.basename(source).find('User')+4]
        else:
            pattern=site+'.lidar.'+z_id+'.'+'00'+'.'+'\d{8}.\d{6}.(stare|user)\d{1}'+os.path.splitext(source)[1]
            match = re.search(pattern, os.path.basename(source))
            if match is not None:
                self.logger.log(f"{source} will not be renamed.")
                return source
            else:
                self.logger.log(f"Renaming of {source} not supported.")
                return
            
        match = re.search(pattern, os.path.basename(source))
        date_part = match.group(1)  
        if len(match.group(2))<6:
            with open(source, "r") as f:
                lines = []
                for line_num in range(11):
                    lines.append(f.readline())
                metadata={}
                for line in lines:
                    metaline = line.split(":")
                    if "Start time" in metaline:
                        metadata["Start time"] = metaline[1:]
                    else:
                        metadata[metaline[0]] = metaline[1]  # type: ignore
                time_part = match.group(2)+metadata["Start time"] [1]+metadata["Start time"] [2].split('.')[0]
        else:
            time_part = match.group(2)
            
        save_filename=site+'.lidar.'+z_id+'.'+'00'+'.'+date_part+'.'+time_part+'.'+scan_type+os.path.splitext(source)[1]
        if os.path.exists(os.path.join(save_path,save_filename))==False or replace==True:
            self._copy_raw_file(source, os.path.join(save_path,save_filename))

        return os.path.join(os.path.join(save_path,save_filename))
    
    @with_logging
    def rename_windcube_200s(self,source,site,z_id,save_path=None,replace=False):
        return source
    
    @with_logging
    def read_halo_xr(self,source,overlapping_distance=1.5):
        
        '''
        Format raw Halo XR data file into WDH-compatible netCDF
        
        Inputs:
            source: str
                source of the 00-level file
          
            overlapping_distance: float
                distance between gates in overlapping mode in meters
                
        Outputs:
            outputData: netDFC
                formatted data structure
        '''
        
        with open(source, "r") as f:
            lines = []
            for line_num in range(11):
                lines.append(f.readline())
    
            # read metadata into strings
            metadata = {}
            for line in lines:
                metaline = line.split(":")
                if "Start time" in metaline:
                    metadata["Start time"] = metaline[1:]
                else:
                    metadata[metaline[0]] = metaline[1]  # type: ignore
    
            # convert some metadata
            num_gates = int(metadata["Number of gates"])  # type: ignore
    
            # Read some of the label lines
            for line_num in range(6):
                f.readline()
    
            # initialize arrays
            time = []
            azimuth = []
            elevation = []
            pitch = []
            roll = []
            doppler = []
            intensity = []
            beta = []
            
            while True:
                a = f.readline().split()
                if not len(a):  # is empty
                    break
    
                time.append(float(a[0]))
                azimuth.append(float(a[1]))
                elevation.append(float(a[2]))
                pitch.append(float(a[3]))
                roll.append(float(a[4]))
    
                doppler.append([0] * num_gates)
                intensity.append([0] * num_gates)
                beta.append([0] * num_gates)
                
                for _ in range(num_gates):
                    b = f.readline().split()
                    range_gate = int(b[0])
                    doppler[-1:][0][range_gate] = float(b[1])
                    intensity[-1:][0][range_gate] = float(b[2])
                    beta[-1:][0][range_gate] = float(b[3])
    
        # convert date to np.datetime64
        start_time_string = "{}-{}-{}T{}:{}:{}".format(
            metadata["Start time"][0][1:5],  # year
            metadata["Start time"][0][5:7],  # month
            metadata["Start time"][0][7:9],  # day
            "00",  # hour
            "00",  # minute
            "00.00",  # second
        )
    
        # find times where it wraps from 24 -> 0, add 24 to all indices after
        time=np.array(time)
        new_day_indices = np.where(np.diff(time) < -23)
        for new_day_index in new_day_indices[0]:
            time[new_day_index + 1 :] += 24.0
    
        start_time = np.datetime64(start_time_string)
        datetimes = [
            start_time + np.timedelta64(int(3600 * 1e9 * dtime), "ns") for dtime in time
        ]
    
        outputData = xr.Dataset(
            {
                # "time": (("time"), time),
                "azimuth": (("time"), azimuth),
                "elevation": (("time"), elevation),
                "pitch": (("time"), pitch),
                "roll": (("time"), roll),
                "wind_speed": (("time", "range_gate"), doppler),
                "intensity": (("time", "range_gate"), intensity),
                "beta": (("time", "range_gate"), beta),
            },
            coords={"time": np.array(datetimes), "range_gate": np.arange(num_gates)},
            attrs={"Range gate length (m)": float(metadata["Range gate length (m)"])},  # type: ignore
        )
    
        # Save some attributes
        outputData.attrs["Range gate length (m)"] = float(
            metadata["Range gate length (m)"]  # type: ignore
        )
        outputData.attrs["Number of gates"] = float(metadata["Number of gates"])  # type: ignore
        outputData.attrs["Scan type"] = str(metadata["Scan type"]).strip()
        outputData.attrs["Pulses per ray"] = float(metadata["Pulses/ray"])  # type: ignore
        outputData.attrs["System ID"] = str(metadata["System ID"]).strip()  # type: ignore
        outputData.attrs["source"] = str(metadata["Filename"])[1:-1]
        outputData.attrs["code_version"]=''
        outputData.attrs["title"]='Lidar Halo XR'
        outputData.attrs["description"]='AWAKEN XR Halo Lidar data'
        outputData.attrs["location_id"]=os.path.basename(source).split('.')[0]
        outputData.attrs["latitude"]=self.config.latitude
        outputData.attrs["longitude"]=self.config.longitude
        outputData.attrs["altitude"]=self.config.altitude
        outputData.attrs["site"]=self.config.site_description
        
        outputData["distance"] = (
            "range_gate",
            outputData.coords["range_gate"].data * outputData.attrs["Range gate length (m)"]
            + outputData.attrs["Range gate length (m)"] / 2,
        )
        outputData["distance_overlapped"] = (
            "range_gate",
            outputData.coords["range_gate"].data * overlapping_distance
            + outputData.attrs["Range gate length (m)"] / 2,
        )
        intensity = outputData.intensity.data.copy()
        intensity[intensity <= 1] = np.nan
        outputData['SNR']=xr.DataArray(data=10 * np.log10(intensity - 1),coords={"time": np.array(datetimes), "range_gate": np.arange(num_gates)})
    
        # Dynamically add scan type and z-id (z02, z03, etc) to outputData metadata
        # loc_id, instrument, z02/z03, data '00', date, time, scan type, extension
        raw_basename = source.replace("\\", "/").rsplit("/")[-1]
        if ".z" in raw_basename:
            _, _, z_id, _, _, _, scan_type, _ = raw_basename.lower().split(".")
        else:  # local NREL tsdat-ing
            z_id = str(outputData.attrs["System ID"])
            scan_type = ""
            if "user" in raw_basename.lower():
                scan_type = raw_basename.split("_")[0].lower()
            elif "stare" in raw_basename.lower():
                scan_type = "stare"
            elif "vad" in raw_basename.lower():
                scan_type = "vad"
            elif "wind_profile" in raw_basename.lower():
                scan_type = "wind_profile"
            elif "rhi" in raw_basename.lower():
                scan_type = "rhi"
    
        valid_types = ["user", "stare", "vad", "wind_profile", "rhi"]
        if not any(valid_type in scan_type for valid_type in valid_types):
            self.logger.log(f"Scan type '{scan_type}' not supported.")
    
        outputData.attrs["scan_type"] = scan_type
        outputData.attrs["z_id"] = z_id
            
        #sort data
        outputData=outputData.sortby('time')
        return outputData
    
    @with_logging
    def read_windcube_200s(self, source):
       
        #load data
        temp = xr.load_dataset(source)
        group_name = temp["sweep_group_name"].data[0]
        outputData = xr.load_dataset(source, group=group_name, decode_times=False)

        #format time
        outputData = outputData.assign_coords(time=np.array([datetime.utcfromtimestamp(t) for t in outputData.time.data]))

        #rename variables
        outputData=outputData.rename({'radial_wind_speed':'wind_speed','cnr':'SNR','gate_index':'range_gate'})
        
        #replace range with range_gate 
        outputData=outputData.set_coords('range_gate')
        outputData["range_gate"] = outputData["range_gate"].drop_vars("range").swap_dims({"range": "range_gate"})
        for v in ["wind_speed",
            "atmospherical_structures_type",
            "SNR",
            "doppler_spectrum_mean_error",
            "doppler_spectrum_width",
            "radial_wind_speed_ci",
            "radial_wind_speed_status"]:
        
            outputData[v]=outputData[v].drop_vars("range").swap_dims({"range": "range_gate"})
        
        
        #save range as distance
        distance=outputData['range']
        outputData['distance']=distance

        #dd missing variables
        outputData['pitch']=outputData['azimuth']*0
        outputData['roll']=outputData['azimuth']*0
        
        #add missing attributes
        outputData.attrs['Scan type']=''
        outputData.attrs['code_version']=''
        outputData.attrs['location_id']=os.path.basename(source).split('.')[0]
        
        #delete stuff
        outputData = outputData.drop_vars("range")
        del outputData["time_reference"]
        bad_encodings = ["szip", "zstd", "bzip2", "blosc"]
        for e in bad_encodings:
            for var in outputData.keys():
                if e in outputData[var].encoding:
                    del outputData[var].encoding[e]

        return outputData

    def plot_raw_data(self,data,save_figures=True):
        
        # make colormap of range gate vs time colored with wind speed
        fig1, ax1 = plt.subplots(figsize=(18, 8))
        data.wind_speed.plot(
            ax=ax1,
            x="time",
            cmap="coolwarm",
            vmin=-15,
            vmax=15,
            cbar_kwargs={'label': r"Radial wind speed [m s$^{-1}$]"}
        )
        
        date=str(data.time.values[0])[:10]

        fig1.suptitle(
            f"Radial wind speed at {self.config.site} on {date} \n File: "
            + os.path.basename(self.source)
        )

        format_time_xticks(ax1)
        ax1.set_xlabel("Time (UTC)")
        ax1.set_ylabel("Range gate")

        if save_figures:
            fig1.savefig(self.save_filename.replace('nc','wind_speed_v_dist_time.png'))
            plt.close(fig1)

        # make colormap of range gate vs time colored with SNR
        fig2, ax2 = plt.subplots(figsize=(18, 8))

        data.SNR.plot(
            ax=ax2,
            x="time",
            cmap="coolwarm",
            vmin=-30,
            vmax=0,
            cbar_kwargs={'label': "SNR [dB]"}
        )

        fig2.suptitle(
            f"Signal-to-noise ratio at {self.config.site} on {date} \n File: "
            + os.path.basename(self.source)
        )

        format_time_xticks(ax2)
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel("Range gate")

        if save_figures:
            fig2.savefig(self.save_filename.replace('nc','snr_v_dist_time.png'))
            plt.close(fig2)


    
    