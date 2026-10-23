import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import socket
from typing import Union, Optional
import matplotlib as mpl
import getpass
from .logger import SingletonLogger
from functools import wraps
from dataclasses import is_dataclass, fields, asdict
import matplotlib.pyplot as plt
from lidargo.config import LidarConfigFormat,LidarConfigStand
import re

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


def lidar_xyz(r, ele, azi):
    """
    Convert spherical to Cartesian coordinates
    """
    R = r
    A = np.pi / 2 - np.radians(azi)
    E = np.radians(ele)

    X = R * np.cos(E) * np.cos(A)
    Y = R * np.cos(E) * np.sin(A)
    Z = R * np.sin(E)

    return X, Y, Z


def dropDuplicatedCoords(ds, varsToClean=["x", "y", "z"]):
    """drop duplicated Cartesian coordinate info"""
    dupcoord = []
    for coord in ds.coords:
        tmp = ds.x.std(dim=coord).mean()
        if tmp < 1e-10:
            dupcoord.append(coord)

    cleanCoords = ds[varsToClean].mean(dupcoord)
    return ds.assign(cleanCoords)


def defineLocalBins(df, config):
    """
    Helper function for making tidy bins based on ranges and bin sizes
    """

    def bins(coord, delta):
        return np.arange(
            np.floor(coord.min()) - delta / 2,
            np.ceil(coord.max()) + delta,
            delta,
        )

    keys = [field.name for field in fields(config)]

    if "dx" in keys:
        df["xbins"] = pd.cut(df.x, bins(df.x, config.dx))
    if "dy" in keys:
        df["ybins"] = pd.cut(df.y, bins(df.y, config.dy))
    if "dz" in keys:
        df["zbins"] = pd.cut(df.z, bins(df.z, config.dz))
    if "dtime" in keys:
        df["timebins"] = pd.cut(df.deltaTime, bins(df.deltaTime, config.dtime))

    return df


def mid(x):
    """
    Mid point in vector
    """
    return (x[:-1] + x[1:]) / 2


def rev_mid(x):
    '''
    Given midpoint, build edges
    '''
    return np.concatenate([[x[0]-(x[1]-x[0])/2],(x[:-1]+x[1:])/2,[x[-1]+(x[-1]-x[-2])/2]])


def gaussian(x, sigma):
    """
    Gaussian function
    """
    return np.exp(-(x**2) / (2 * sigma**2))


def datestr(num, format="%Y-%m-%d %H:%M:%S.%f"):
    """
    Unix time to string of custom format
    """
    from datetime import datetime

    string = datetime.utcfromtimestamp(num).strftime(format)
    return string


def dt64_to_num(dt64):
    """
    numpy.datetime64[ns] time to Unix time
    """
    tnum = (dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return tnum


def floor(value, step):
    """
    A helper method for floor operation
    """
    return np.floor(value / step) * step


def ceil(value, step):
    """
    A helper method for ceil operation
    """
    return np.ceil(value / step) * step

def nanmin_time(value,_format='%Y-%m-%d %H:%M:%S'):
    min_time=np.min(value[value>np.datetime64('1970-01-01T00:00:00')])
    return min_time.dt.strftime(_format).values


def nanmax_time(value,_format):
    max_time=np.max(value[value>np.datetime64('1970-01-01T00:00:00')])
    return max_time.dt.strftime(_format).values

def remove_labels(fig):
    """
    Graphic helper
    """
    axs = fig.axes

    for ax in axs:
        try:
            loc = ax.get_subplotspec()

            if loc.is_last_row() == False:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            if loc.is_first_col() == False:
                ax.set_yticklabels([])
                ax.set_ylabel("")
        except:
            pass
        
def same_axis(axs):
    # Get the global limits for all subplots
    all_x_limits = [ax.get_xlim() for ax in axs]
    all_y_limits = [ax.get_ylim() for ax in axs]
    
    global_xlim = (min(l[0] for l in all_x_limits), max(l[1] for l in all_x_limits))
    global_ylim = (min(l[0] for l in all_y_limits), max(l[1] for l in all_y_limits))
    
    # Set the same limits for all subplots
    for ax in axs:
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)


def add_attributes(ds):
    """Add standardized attributes to output data variables and dataset."""

    # Remove existing ancillary variables
    for v in ds.variables:
        if "ancillary_variables" in ds[v].attrs:
            del ds[v].attrs["ancillary_variables"]
        # Set standard_name for all variables
        ds[v].attrs["standard_name"] = v

    # Define common attributes for multiple variables
    common_attrs = {
        "azimuth": {
            "long_name": "Azimuth",
            "description": "Beam azimuth angle",
            "units": "degrees",
        },
        "elevation": {
            "long_name": "Elevation",
            "description": "Beam elevation angle",
            "units": "degrees",
        },
        "pitch": {
            "long_name": "Pitch",
            "description": "Lidar pitch angle",
            "units": "degrees",
        },
        "roll": {
            "long_name": "Roll",
            "description": "Lidar roll angle",
            "units": "degrees",
        },
        "range": {
            "long_name": "Range",
            "description": "Distance from the lidar source.",
            "units": "m",
        },
        "beamID": {
            "long_name": "Beam ID",
            "description": "Index of the beam within a scan.",
            "units": "int",
        },
        "scanID": {
            "long_name": "Scan ID",
            "description": "Repetition index of the scan.",
            "units": "int",
        },
        "x": {
            "long_name": "x-direction",
            "description": "x-direction.",
            "units": "m",
        },
        "y": {
            "long_name": "y-direction",
            "description": "y-direction.",
            "units": "m",
        },
        "z": {
            "long_name": "z-direction",
            "description": "z-direction.",
            "units": "m",
        },
        "wind_speed": {
            "long_name": "Line-of-sight velocity",
            "description": "Line-of-sight velocity.",
            "units": "m/s",
            "ancilary_variables": "qc_wind_speed",
        },
        "SNR": {
            "long_name": "Signal-to-noise ratio",
            "description": "Signal-to-noise ratio.",
            "units": "dB",
        },
        "beta": {
            "long_name": "Attenuated backscatter coefficient",
            "description": "Attenuated backscatter coefficient.",
            "units": "1/(m sr)",
            "ancilary_variables": "qc_wind_speed",
        },
        "rws_norm": {
            "long_name": "Normalized radial wind speed",
            "description": "Fluctuation of radial wind speed on top of the binned spatio-temporal median. It is used in the dynamic filter.",
            "units": "m/s",
        },
        "snr_norm": {
            "long_name": "Normalized SNR",
            "description": "Fluctuation of signal-to-noise ratio on top of the binned spatio-temporal median. It is used in the dynamic filter.",
            "units": "dB",
        },
        "probability": {
            "long_name": "Probability",
            "description": "Value of 2-D p.d.f. in the rws_norm vs snr_norm plane. It is used in the dynamic filter.",
            "units": "no units",
        },
        "time": {
            "long_name": "Time UTC",
            "description": "Timestamp in UTC format for the specific beam.",
        },
    }

    # Apply common attributes
    for var, attrs in common_attrs.items():
        if var in ds:
            ds[var].attrs.update(attrs)

    return ds


def add_qc_attrs(ds, qcAttrDict: dict):
    # Set up QC wind speed attributes
    qc_attrs = {
        "units": "int",
        "long_name": "Wind speed QC flag",
        "description": "This variable contains bit-packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits.",
        "bit_0_description": "Value retained.",
        "bit_0_assessment": "Good",
    }

    ds["qc_wind_speed"].attrs.update(qc_attrs)

    # Add QC flag descriptions
    for k, v in qcAttrDict["qc_flag"].items():
        ds["qc_wind_speed"].attrs[
            f"bit_{v}_description"
        ] = f"Value rejected due to {k} criterion."
        ds["qc_wind_speed"].attrs[f"bit_{v}_assessment"] = "Bad"

    # Add config attributes to QC wind speed
    if is_dataclass(qcAttrDict["processConfig"]):
        qcAttrDict["processConfig"] = asdict(qcAttrDict["processConfig"])

    ds["qc_wind_speed"].attrs.update(qcAttrDict["processConfig"])
    ds["qc_wind_speed"].attrs.update(
        {
            "qc_probability_threshold": qcAttrDict["qc_probability_threshold"],
        }
    )

    # Set global attributes
    global_attrs = {
        "data_level": qcAttrDict["data_level"],
        "datastream": os.path.basename(qcAttrDict["datastream"]),
        "input_source": os.path.basename(qcAttrDict["input_source"]),
        "contact": "stefano.letizia@nlr.gov",
        "institution": "NLR",
        "description": "Halo XR/XR+ Lidar standardized and quality-controlled data",
        "history": (
            f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using {os.path.basename(sys.argv[0])}"
        ),
    }

    ds.attrs.update(global_attrs)
    # if "code_version" in ds.attrs:
    #     del ds.attrs["code_version"]

    return ds

def format_time_xticks(
    ax: plt.Axes,
    start: int = 4,
    stop: int = 21,
    step: int = 4,
    date_format: str = "%H:%M",
):
    """----------------------------------------------------------------------------
    Formats the ticks on the x-axis of the provided `plt.Axes` nicely. Requires the
    provided `plt.Axis` to already have a plot attached to it and for the x-axis of
    the plotted data to be a datetime object (numpy / pandas / xarray OK). Sets
    major tick locations by hour according to the provided `start`, `stop`, and
    `step` parameters, and labels ticks according to the provided `date_format`.
    Has nice defaults for a plot spanning a 24-hour period.

    Args:
        ax (plt.Axes): The handle for the axes object on which to format the ticks.
        start (int, optional): Hour in which to start the xticks. Defaults to 4.
        stop (int, optional): Hour in which to stop the xticks. Defaults to 21.
        step (int, optional): The step in between major xticks. Defaults to 4.
        date_format (str, optional): The format to use for xtick labels. Defaults
        to "%H:%M".

    ----------------------------------------------------------------------------"""
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(byhour=range(start, stop, step)))  # type: ignore
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(date_format))  # type: ignore
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
    

def _load_configuration(config: Union[str, dict, LidarConfigFormat, LidarConfigStand],source=None, level:str='standardize'):
    """
    Load configuration from either a file path, dictionary, or LidarConfig object.

    Args:
        config (str, dict, or LidarConfig): Configuration source

    Returns:
        LidarConfig or None: Configuration parameters or None if loading fails
    """
    try:
        if isinstance(config, LidarConfigFormat) and level=='format':
            return config, "Configuration successfully loaded"
        if isinstance(config, LidarConfigStand) and level=='standardize':
            return config, "Configuration successfully loaded"
        elif isinstance(config, str):
            return _load_config_from_file(config,source,level)
        elif isinstance(config, dict):
            if level=='standardize':
                return LidarConfigStand(**config), "Configuration successfully loaded"
            elif level=='format':
                return LidarConfigFormat(**config), "Configuration successfully loaded"
            else:
                return None, f"{level} is an invalid configuration level (must be standardize or format)"
        else:
            return None, f"Invalid config type. Expected str, dict, or LidarConfig, got {type(config)}"
            
    except Exception as e:
        return None, f"Error loading configuration: {str(e)}"


def _load_config_from_file(config_file: str, source: str, level: str):
    """
    Load configuration from an Excel file.

    Args:
        config_file (str): Path to Excel configuration file

    Returns:
        LidarConfig or None: Configuration parameters or None if loading fails
    """
    configs = pd.read_excel(config_file,header=None).set_index(0)
    date_source = np.int64(re.search(r"\d{8}", source).group(0))

    matches = []
    for c in configs.columns:
        regex=configs[c]['regex']
        if "start_date" not in  configs[c]:
            sdate=19700101
        else:
            sdate = configs[c]["start_date"]
        if "end_date" not in  configs[c]:
            edate=30000101
        else:
            edate = configs[c]["end_date"]
        
        match = re.findall(regex, source)
        if len(match) > 0 and sdate <= date_source <= edate:
            matches.append(c)

    if not matches:
        return None,"No regular expression/date range matching the file name"
        
    elif len(matches) > 1:
        return None, "Multiple regular expressions/date ranges matching the file name"

    config_dict = configs[matches[0]].to_dict()
    try:
        if level=='standardize':
            return LidarConfigStand(**config_dict), "Configuration successfully loaded"
        elif level=='format':
            return LidarConfigFormat(**config_dict), "Configuration successfully loaded"
        else:
            return None, f"{level} is an invalid configuration level (must be standardize or format)"
    except Exception as e:
        return None, f"Error validating configuration: {str(e)}"
