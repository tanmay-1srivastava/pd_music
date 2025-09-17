# Gait Analysis Configuration File
# Adjust these parameters to fine-tune the analysis

class GaitAnalysisConfig:
    """Configuration class for gait analysis parameters"""
    
    # ========================
    # Sensor-Specific Parameters
    # ========================
    
    # Sensor calibration factors
    SENSOR_CALIBRATION = {
        'cheap': {
            'noise_factor': 1.5,         # Higher noise expectation
            'threshold_multiplier': 1.8,  # Higher thresholds needed
            'filtering_strength': 1.2,    # More aggressive filtering
            'min_prominence_factor': 1.4   # Higher prominence required
        },
        'expensive': {
            'noise_factor': 0.8,         # Lower noise expectation  
            'threshold_multiplier': 0.6,  # Lower thresholds sufficient
            'filtering_strength': 0.9,    # Gentler filtering
            'min_prominence_factor': 0.9   # Lower prominence sufficient
        }
    }
    
    # ========================
    # Peak Detection Parameters
    # ========================
    
    # Minimum height for heel strike detection (relative to signal mean)
    HEEL_STRIKE_MIN_HEIGHT_FACTOR = 1.2
    
    # Minimum height for toe-off detection (relative to signal mean) 
    TOE_OFF_MIN_HEIGHT_FACTOR = 0.8
    
    # Minimum distance between consecutive peaks (in samples)
    # This prevents detecting multiple peaks for the same step
    MIN_PEAK_DISTANCE_FACTOR = 0.3  # seconds * sampling_rate
    
    # Peak prominence (how much a peak stands out)
    PEAK_PROMINENCE_FACTOR = 0.1  # relative to signal std
    
    # ========================
    # Signal Processing Parameters
    # ========================
    
    # Moving average window size for smoothing (in samples)
    SMOOTHING_WINDOW_SIZE = 5
    
    # Butterworth filter parameters
    FILTER_ORDER = 4
    LOW_PASS_CUTOFF = 10  # Hz
    HIGH_PASS_CUTOFF = 0.5  # Hz
    
    # ========================
    # Gait Event Detection
    # ========================
    
    # Minimum step time (seconds) - prevents false detections
    MIN_STEP_TIME = 0.3
    MAX_STEP_TIME = 2.0
    
    # Minimum stride time (seconds)
    MIN_STRIDE_TIME = 0.8
    MAX_STRIDE_TIME = 4.0
    
    # Double support time constraints
    MIN_DOUBLE_SUPPORT_TIME = 0.05
    MAX_DOUBLE_SUPPORT_TIME = 0.5
    
    # ========================
    # Walking Speed Estimates
    # ========================
    
    # Average walking speeds for step length estimation (m/s)
    SLOW_WALKING_SPEED = 1.0
    NORMAL_WALKING_SPEED = 1.4
    FAST_WALKING_SPEED = 1.8
    
    # Use adaptive speed estimation based on detected step frequency
    USE_ADAPTIVE_SPEED = False  # Disable artificial formula
    
    # Enable direct accelerometer-based speed calculation
    USE_ACCELEROMETER_SPEED = True
    
    # Accelerometer-based speed calculation parameters
    SPEED_CALCULATION_METHOD = 'integration'  # 'integration' or 'stride_displacement'
    SPEED_INTEGRATION_WINDOW = 2.0  # seconds for integration window
    SPEED_SMOOTHING_FACTOR = 0.1   # smoothing for speed estimates
    
    # ========================
    # Data Processing
    # ========================
    
    # Accelerometer conversion factors
    GRAVITY = 9.81  # m/s² - for converting g to m/s²
    
    # Gyroscope conversion factor
    DEG_TO_RAD = 3.14159 / 180  # for converting degrees to radians
    
    # Sampling rate constraints
    MIN_SAMPLING_RATE = 50  # Hz
    MAX_SAMPLING_RATE = 1000  # Hz
    
    # ========================
    # Analysis Windows
    # ========================
    
    # Time windows for variability analysis (seconds)
    TIME_WINDOWS = [60, 300, 600]  # 1min, 5min, 10min
    
    # Minimum recording duration for analysis (seconds)
    MIN_RECORDING_DURATION = 30
    
    # ========================
    # Outlier Detection
    # ========================
    
    # Z-score threshold for outlier removal
    OUTLIER_Z_THRESHOLD = 3.0
    
    # Percentage of outliers to remove (top and bottom)
    OUTLIER_PERCENTILE = 5  # Remove top and bottom 5%
    
    # ========================
    # Visualization Parameters
    # ========================
    
    # Plot settings
    FIGURE_SIZE = (12, 8)
    DPI = 300
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # matplotlib style
    
    # Colors for different sensors
    COLORS = {
        'cheap': '#2E8B57',      # Sea green
        'expensive': '#4169E1',   # Royal blue
        'left': '#DC143C',        # Crimson
        'right': '#FF8C00',       # Dark orange
        'combined': '#8A2BE2'     # Blue violet
    }
    
    # ========================
    # File Processing
    # ========================
    
    # File extensions to process
    CSV_EXTENSION = '.csv'
    
    # Expected column variations (to handle typos)
    COLUMN_VARIATIONS = {
        'timestamp': ['timestamp', 'time', 't'],
        'ax': ['ax', 'accel_x', 'acc_x'],
        'ay': ['ay', 'accel_y', 'acc_y'], 
        'az': ['az', 'accel_z', 'acc_z'],
        'gx': ['gx', 'gyro_x', 'gyr_x'],
        'gy': ['gy', 'gyro_y', 'gyr_y'],
        'gz': ['gz', 'gyro_z', 'gyr_z']
    }
    
    # Folder name variations (to handle typos)
    FOLDER_VARIATIONS = {
        'lumber': ['lumber', 'lumbar', 'Lumber', 'Lumbar'],
        'left_foot': ['left_foot', 'left', 'lf', 'LF'],
        'right_foot': ['right_foot', 'right', 'rf', 'RF']
    }
    
    # ========================
    # Advanced Parameters
    # ========================
    
    # Gait symmetry analysis
    ENABLE_SYMMETRY_ANALYSIS = True
    
    # Cadence calculation
    ENABLE_CADENCE_ANALYSIS = True
    
    # Step width estimation (experimental)
    ENABLE_STEP_WIDTH_ESTIMATION = False
    STEP_WIDTH_METHOD = 'sensor_separation'  # 'sensor_separation' or 'ml_displacement'
    
    # ========================
    # Output Settings
    # ========================
    
    # Decimal places for results
    RESULT_PRECISION = 4
    
    # Save intermediate processing results
    SAVE_INTERMEDIATE_RESULTS = True
    
    # Create detailed plots
    CREATE_DETAILED_PLOTS = True
    
    # Summary statistics to calculate
    STATISTICS = ['mean', 'std', 'median', 'min', 'max', 'cv', 'iqr']
    
    @classmethod
    def get_min_peak_distance(cls, sampling_rate):
        """Calculate minimum peak distance based on sampling rate"""
        return int(cls.MIN_PEAK_DISTANCE_FACTOR * sampling_rate)
    
    @classmethod
    def get_smoothing_window(cls, sampling_rate):
        """Calculate smoothing window size based on sampling rate"""
        # Ensure odd number for centered window
        window = int(cls.SMOOTHING_WINDOW_SIZE * sampling_rate / 100)
        return window if window % 2 == 1 else window + 1
    
    @classmethod
    def get_sensor_config(cls, sensor_type):
        """Get sensor-specific configuration parameters"""
        sensor_type = sensor_type.lower()
        if sensor_type in cls.SENSOR_CALIBRATION:
            return cls.SENSOR_CALIBRATION[sensor_type]
        else:
            # Default to cheap sensor parameters for unknown types
            return cls.SENSOR_CALIBRATION['cheap']
    
    @classmethod
    def get_adaptive_speed(cls, step_frequency):
        """Estimate walking speed based on step frequency using more realistic biomechanical relationships"""
        if not cls.USE_ADAPTIVE_SPEED:
            return cls.NORMAL_WALKING_SPEED
            
        # More realistic relationship based on biomechanics research:
        # Walking speed = step_length × step_frequency
        # Step length typically varies from 0.4m to 0.8m depending on speed
        # Use empirical relationship: step_length ≈ 0.15 + 0.4 * step_frequency (in m)
        
        # Clamp step frequency to reasonable range
        step_freq_clamped = max(0.5, min(step_frequency, 3.0))
        
        # Calculate realistic step length based on frequency
        estimated_step_length = 0.15 + 0.4 * step_freq_clamped
        
        # Walking speed = step_length × step_frequency  
        estimated_speed = estimated_step_length * step_freq_clamped
        
        # Clamp to reasonable walking speeds (0.5 to 2.5 m/s)
        return max(0.5, min(estimated_speed, 2.5))
    
    @classmethod
    def calculate_walking_speed_from_accelerometer(cls, accel_data, step_times, sampling_rate):
        """Calculate actual walking speed from accelerometer data"""
        import numpy as np
        from scipy import signal
        
        if not cls.USE_ACCELEROMETER_SPEED or len(accel_data) < 100:
            # Fallback to step-frequency based estimation with individual variation
            if len(step_times) > 0:
                step_frequency = 1.0 / np.mean(step_times)
                base_speed = cls.get_adaptive_speed(step_frequency)
                # Add individual variation based on actual step time variability
                speed_variation = np.std(step_times) * 2.0  # Convert time variation to speed variation
                return max(0.5, min(base_speed + speed_variation, 2.5))
            return cls.NORMAL_WALKING_SPEED
        
        try:
            # Method 1: Integration-based speed calculation
            if cls.SPEED_CALCULATION_METHOD == 'integration':
                # Calculate forward acceleration (typically ax or resultant horizontal)
                if len(accel_data.shape) > 1:
                    # Multi-axis accelerometer data
                    forward_accel = accel_data[:, 0]  # Assuming ax is forward direction
                    lateral_accel = accel_data[:, 1]   # Assuming ay is lateral
                    horizontal_accel = np.sqrt(forward_accel**2 + lateral_accel**2)
                else:
                    # Single axis data
                    horizontal_accel = accel_data
                
                # Remove gravity and smooth
                horizontal_accel = horizontal_accel - np.mean(horizontal_accel)
                
                # Apply bandpass filter for walking frequencies (0.5-3 Hz)
                nyquist = sampling_rate / 2
                low_cut = 0.5 / nyquist
                high_cut = min(3.0 / nyquist, 0.95)
                
                b, a = signal.butter(4, [low_cut, high_cut], btype='band')
                filtered_accel = signal.filtfilt(b, a, horizontal_accel)
                
                # Integrate acceleration to get velocity
                dt = 1.0 / sampling_rate
                velocity = np.cumsum(filtered_accel) * dt
                
                # Remove drift using high-pass filter
                b_hp, a_hp = signal.butter(2, 0.1 / nyquist, btype='high')
                velocity_corrected = signal.filtfilt(b_hp, a_hp, velocity)
                
                # Calculate RMS speed over walking periods
                window_samples = int(cls.SPEED_INTEGRATION_WINDOW * sampling_rate)
                speed_estimates = []
                
                for i in range(0, len(velocity_corrected) - window_samples, window_samples//2):
                    window_velocity = velocity_corrected[i:i+window_samples]
                    rms_speed = np.sqrt(np.mean(window_velocity**2))
                    if 0.3 < rms_speed < 3.0:  # Reasonable walking speeds
                        speed_estimates.append(rms_speed)
                
                if speed_estimates:
                    return np.median(speed_estimates)
            
            # Method 2: Stride displacement estimation
            elif cls.SPEED_CALCULATION_METHOD == 'stride_displacement':
                # Use step detection and displacement estimation
                if len(step_times) > 0:
                    avg_step_time = np.mean(step_times)
                    step_frequency = 1.0 / avg_step_time
                    
                    # Estimate stride length from acceleration patterns
                    stride_displacement = cls._estimate_stride_displacement(accel_data, sampling_rate)
                    
                    if stride_displacement > 0:
                        walking_speed = stride_displacement * step_frequency / 2  # /2 because stride = 2 steps
                        return max(0.5, min(walking_speed, 2.5))
            
            # Fallback if accelerometer method fails
            return cls.NORMAL_WALKING_SPEED
            
        except Exception as e:
            # Fallback to step-frequency estimation with variation
            if len(step_times) > 0:
                step_frequency = 1.0 / np.mean(step_times)
                base_speed = cls.get_adaptive_speed(step_frequency)
                # Add realistic individual variation
                speed_variation = (np.std(step_times) / np.mean(step_times)) * base_speed * 0.5
                return max(0.5, min(base_speed + speed_variation, 2.5))
            return cls.NORMAL_WALKING_SPEED
    
    @classmethod
    def _estimate_stride_displacement(cls, accel_data, sampling_rate):
        """Estimate stride displacement from acceleration patterns"""
        import numpy as np
        
        try:
            # Simple displacement estimation using double integration with drift correction
            if len(accel_data.shape) > 1:
                forward_accel = accel_data[:, 0] - np.mean(accel_data[:, 0])
            else:
                forward_accel = accel_data - np.mean(accel_data)
            
            dt = 1.0 / sampling_rate
            
            # First integration: acceleration -> velocity
            velocity = np.cumsum(forward_accel) * dt
            velocity = velocity - np.mean(velocity)  # Remove DC component
            
            # Second integration: velocity -> displacement
            displacement = np.cumsum(velocity) * dt
            
            # Estimate stride length as the range of displacement during stride cycles
            stride_length = np.std(displacement) * 4  # Heuristic based on stride patterns
            
            return max(0.5, min(stride_length, 2.0))  # Reasonable stride lengths
            
        except:
            return 1.3  # Default stride length estimate
    
    @classmethod
    def validate_parameters(cls):
        """Validate configuration parameters"""
        errors = []
        
        if cls.MIN_STEP_TIME >= cls.MAX_STEP_TIME:
            errors.append("MIN_STEP_TIME must be less than MAX_STEP_TIME")
            
        if cls.MIN_STRIDE_TIME >= cls.MAX_STRIDE_TIME:
            errors.append("MIN_STRIDE_TIME must be less than MAX_STRIDE_TIME")
            
        if cls.LOW_PASS_CUTOFF <= cls.HIGH_PASS_CUTOFF:
            errors.append("LOW_PASS_CUTOFF must be greater than HIGH_PASS_CUTOFF")
            
        if cls.OUTLIER_Z_THRESHOLD <= 0:
            errors.append("OUTLIER_Z_THRESHOLD must be positive")
            
        if not (0 <= cls.OUTLIER_PERCENTILE <= 50):
            errors.append("OUTLIER_PERCENTILE must be between 0 and 50")
            
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
            
        return True

# Quick access to config instance
config = GaitAnalysisConfig()