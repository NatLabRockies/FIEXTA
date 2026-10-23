from dataclasses import dataclass
from datetime import datetime
import re
import numpy as np

@dataclass
class LidarConfigFormat:
    """Configuration parameters for LIDAR formatting."""
    regex: str=''
    model: str = 'halo'
    site: str = 'sc1'
    instrument_id: int = 1
    data_level_out: str ='a0'
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    site_description: str = 'AWAKEN site C1a'
    
    def _validate_model(self, model: str, field_name: str) -> None:
        """Validate lidal model."""
        if model != 'halo' and model !='windcube':
            raise ValueError(f"Lidar model {model} not supported")
            
    def _validate_z_id(self, instrument_id: str, field_name: str) -> None:
        """Validate instrument ID."""
        if instrument_id<0 or instrument_id>99:
            raise ValueError("Instrument ID must be within 0 and 99")
        
    #Validate data levels
    valid_data_levels = ["a"+str(i) for i in range(10)]
    if data_level_out not in valid_data_levels:
        raise ValueError(f"data_level_out must be one of {valid_data_levels}")

    def validate(self) -> None:
        """Validate all configuration parameters."""
        # Validate dates
        self._validate_model(self.model, "start_date")
        self._validate_z_id(self.instrument_id, "end_date")
        self.z_id=f"z{self.instrument_id:02}"

@dataclass
class LidarConfigStand:
    """Configuration parameters for LIDAR standardization."""
    regex: str=''
    project: str = "raaw"
    name: str = "wake.turb"
    start_date: int = 20220101
    end_date: int = 20240101
    azimuth_offset: float = -90
    min_azi_step: float = -0.1
    max_azi_step: float = 3
    min_ele_step: float = -0.1
    max_ele_step: float = 0.1
    ang_tol: float = 0.25
    count_threshold: float = 0.1
    min_scan_duration: float = 2
    dx: float = 200.0
    dy: float = 200.0
    dz: float = 50.0
    dtime: float = 600.0
    range_min: float = 96.0
    range_max: float = 2000.0
    snr_min: float = -25.0
    N_resonance_bins: int = 50
    max_resonance_rmse: float = 0.1
    rws_max: float = 30.0
    rws_norm_limit: float = 20.0
    rws_standard_error_limit: float = 1
    snr_standard_error_limit: float = 1
    local_population_min_limit: float = 30
    local_scattering_min_limit: float = 0.5
    N_probability_bins: int = 25
    min_percentile: float = 1.0
    max_percentile: float = 99.0
    min_probability_range: float = 0.0001
    max_probability_range: float = 0.5
    rws_norm_increase_limit: float = 0.15
    data_level_in: str = "b0"
    data_level_out: str = "b1"
    ground_level: float = -120.0
    rws_min: float = 0.0
    rename_vars: str = ""
    rename_attrs: str = ""
    range_name: str="distance"
    
    def _validate_date_format(self, date: int, field_name: str) -> None:
        """Validate date format (YYYYMMDD)."""
        date_str = str(date)
        if len(date_str) != 8:
            raise ValueError(f"{field_name} must be in YYYYMMDD format")
        try:
            datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            raise ValueError(f"Invalid {field_name}: {date}")

    def _validate_positive(self, value: float, field_name: str) -> None:
        """Validate that a value is positive."""
        if value <= 0:
            raise ValueError(f"{field_name} must be positive, got {value}")

    def _validate_non_negative(self, value: float, field_name: str) -> None:
        """Validate that a value is non-negative."""
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative, got {value}")

    def _validate_range(
        self, value: float, min_val: float, max_val: float, field_name: str
    ) -> None:
        """Validate that a value falls within an acceptable range."""
        if not min_val <= value <= max_val:
            raise ValueError(
                f"{field_name} must be between {min_val} and {max_val}, got {value}"
            )

    def _validate_string_format(
        self, value: str, pattern: str, field_name: str
    ) -> None:
        """Validate string format against a regex pattern."""
        if not re.match(pattern, value):
            raise ValueError(f"{field_name} format is invalid: {value}")

    def validate(self) -> None:
        """Validate all configuration parameters."""
        # Validate dates
        self._validate_date_format(self.start_date, "start_date")
        self._validate_date_format(self.end_date, "end_date")
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        # Validate project and name format
        if 'str' not in str(type(self.project)):
            if np.isnan(self.project): self.project=''
        self._validate_string_format(self.project, r"^[a-zA-Z0-9_.-]*$", "project")
        
        if 'str' not in str(type(self.name)):
            if np.isnan(self.name): self.name=''
        self._validate_string_format(self.name, r"^[a-zA-Z0-9_.-]*$", "name")

        # Validate angle steps and tolerances
        if self.min_azi_step >= self.max_azi_step:
            raise ValueError("min_azi_step must be less than max_azi_step")
        if self.min_ele_step >= self.max_ele_step:
            raise ValueError("min_ele_step must be less than max_ele_step")
        self._validate_positive(self.ang_tol, "ang_tol")

        # Validate spatial parameters
        self._validate_positive(self.dx, "dx")
        self._validate_positive(self.dy, "dy")
        self._validate_positive(self.dz, "dz")
        self._validate_positive(self.dtime, "dtime")

        # Validate range parameters
        if self.range_min >= self.range_max:
            raise ValueError("range_min must be less than range_max")
        self._validate_positive(self.range_min, "range_min")
        self._validate_positive(self.range_max, "range_max")

        # Validate bin counts
        self._validate_positive(self.N_resonance_bins, "N_resonance_bins")
        self._validate_positive(self.N_probability_bins, "N_probability_bins")

        # Validate percentiles
        if not 0 <= self.min_percentile < self.max_percentile <= 100:
            raise ValueError("Invalid percentile range")

        # Validate probability ranges
        if not 0 <= self.min_probability_range < self.max_probability_range <= 1:
            raise ValueError("Invalid probability range")

        # Validate RWS parameters
        self._validate_positive(self.rws_max, "rws_max")
        self._validate_non_negative(self.rws_min, "rws_min")
        if self.rws_min >= self.rws_max:
            raise ValueError("rws_min must be less than rws_max")
        self._validate_positive(self.rws_norm_limit, "rws_norm_limit")
        self._validate_positive(
            self.rws_standard_error_limit, "rws_standard_error_limit"
        )
        self._validate_range(
            self.rws_norm_increase_limit, 0, 1, "rws_norm_increase_limit"
        )

        # Validate data levels
        valid_data_levels =["a"+str(i) for i in range(10)]+ ["b"+str(i) for i in range(10)]+["c"+str(i) for i in range(10)]
        if self.data_level_in not in valid_data_levels:
            raise ValueError(f"data_level_in must be one of {valid_data_levels}")
        if self.data_level_out not in valid_data_levels:
            raise ValueError(f"data_level_out must be one of {valid_data_levels}")
        if valid_data_levels.index(self.data_level_in) >= valid_data_levels.index(
            self.data_level_out
        ):
            raise ValueError("data_level_out must be higher than data_level_in")

        # Validate threshold parameters
        self._validate_positive(self.count_threshold, "count_threshold")
        self._validate_positive(self.min_scan_duration, "min_scan_duration")
        self._validate_positive(
            self.local_population_min_limit, "local_population_min_limit"
        )
        self._validate_range(
            self.local_scattering_min_limit, 0, 1, "local_scattering_min_limit"
        )
        self._validate_positive(self.max_resonance_rmse, "max_resonance_rmse")
        
                
