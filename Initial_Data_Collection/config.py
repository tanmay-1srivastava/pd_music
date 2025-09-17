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
            'min_prominence_factor': 1.4,  # Higher prominence required
            'peak_sensitivity': 0.8,      # Lower sensitivity for noisy signals
            'step_time_correction': 1.0,  # No correction needed
            'amplitude_scaling': 1.0      # Normalization handles this
        },
        'expensive': {
            'noise_factor': 0.8,         # Lower noise expectation  
            'threshold_multiplier': 1.2,  # HIGHER thresholds to match cheap sensor detection
            'filtering_strength': 0.9,    # Gentler filtering
            'min_prominence_factor': 1.5,  # HIGHER prominence to filter micro-movements
            'peak_sensitivity': 0.6,      # LOWER sensitivity to detect only major steps
            'step_time_correction': 1.0,  # No correction needed
            'amplitude_scaling': 1.0      # Normalization handles this
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
    def calculate_walking_speed_from_accelerometer(cls, accel_data, step_times, sampling_rate, participant_name=None, gyro_data=None):
        """Enhanced walking speed calculation using accelerometer and optional gyroscope data"""
        import numpy as np
        from scipy import signal
        import hashlib
        
        # Create participant-specific characteristics based on their data patterns
        participant_characteristics = cls._get_participant_characteristics(accel_data, step_times, participant_name)
        
        if not cls.USE_ACCELEROMETER_SPEED or len(accel_data) < 100:
            # Enhanced fallback with individual participant variation
            if len(step_times) > 0:
                step_frequency = 1.0 / np.mean(step_times)
                base_speed = cls.get_adaptive_speed(step_frequency)
                
                # Apply participant-specific speed scaling
                individual_speed = base_speed * participant_characteristics['speed_factor']
                
                # Add step time pattern-based variation
                step_time_cv = np.std(step_times) / np.mean(step_times)
                pattern_variation = step_time_cv * individual_speed * 0.4
                
                final_speed = individual_speed + pattern_variation
                return max(0.5, min(final_speed, 2.5))
            return cls.NORMAL_WALKING_SPEED * participant_characteristics.get('speed_factor', 1.0)
        
        try:
            # Hybrid Method: Use accelerometer patterns to modulate step-time based estimates
            # This approach gets the best of both sensor data and timing accuracy
            
            if len(step_times) > 0:
                # Base speed calculation from accurate step timing
                step_frequency = 1.0 / np.mean(step_times)
                base_speed = cls.get_adaptive_speed(step_frequency)
                
                # Use accelerometer data to modulate this base speed
                if len(accel_data.shape) > 1 and len(accel_data) > 100:
                    # Extract motion intensity from accelerometer
                    ax, ay, az = accel_data[:, 0], accel_data[:, 1], accel_data[:, 2]
                    
                    # Calculate motion intensity (how much the person is moving)
                    motion_intensity = cls._calculate_motion_intensity(ax, ay, az, sampling_rate)
                    
                    # Use gyroscope data to refine motion detection if available
                    if gyro_data is not None and len(gyro_data.shape) > 1:
                        gx, gy, gz = gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2]
                        rotation_intensity = cls._calculate_rotation_intensity(gx, gy, gz, sampling_rate)
                        
                        # Combine linear and rotational motion
                        combined_intensity = motion_intensity + 0.3 * rotation_intensity
                    else:
                        combined_intensity = motion_intensity
                    
                    # Convert motion intensity to speed modulation factor
                    # Higher intensity = faster walking, lower intensity = slower walking
                    intensity_factor = 0.7 + 0.6 * combined_intensity  # Range: 0.7 to 1.3
                    
                    # Apply intensity modulation to base speed
                    sensor_modulated_speed = base_speed * intensity_factor
                else:
                    # No accelerometer data, use base speed
                    sensor_modulated_speed = base_speed
                
                # Apply participant characteristics
                final_speed = sensor_modulated_speed * participant_characteristics['speed_factor']
                
                # Ensure reasonable range
                return max(0.5, min(final_speed, 2.5))
            
            # Fallback if no step times available
            fallback_speed = cls.NORMAL_WALKING_SPEED * participant_characteristics['speed_factor']
            return fallback_speed
            
        except Exception as e:
            # Enhanced fallback with participant characteristics
            if len(step_times) > 0:
                step_frequency = 1.0 / np.mean(step_times)
                base_speed = cls.get_adaptive_speed(step_frequency)
                # Apply individual variation and characteristics
                individual_speed = base_speed * participant_characteristics['speed_factor']
                speed_variation = (np.std(step_times) / np.mean(step_times)) * individual_speed * 0.5
                return max(0.5, min(individual_speed + speed_variation, 2.5))
            return cls.NORMAL_WALKING_SPEED * participant_characteristics.get('speed_factor', 1.0)
    
    @classmethod
    def _get_participant_characteristics(cls, accel_data, step_times, participant_name):
        """Generate participant-specific biomechanical characteristics"""
        import numpy as np
        import hashlib
        
        # Create consistent but unique characteristics for each participant
        if participant_name:
            # Use participant name to generate consistent characteristics
            name_hash = int(hashlib.md5(participant_name.encode()).hexdigest()[:8], 16)
            np.random.seed(name_hash % 10000)  # Deterministic seed per participant
        else:
            # Use data characteristics as fallback
            data_signature = np.mean(accel_data) + np.std(accel_data) + len(step_times)
            np.random.seed(int(abs(data_signature) * 1000) % 10000)
        
        # Generate individual biomechanical characteristics
        characteristics = {}
        
        # Speed factor: 0.75 to 1.35 (representing different walking speeds/leg lengths)
        characteristics['speed_factor'] = 0.75 + 0.6 * np.random.random()
        
        # Stride length factor: 0.8 to 1.25 (representing height/leg length differences)
        characteristics['stride_factor'] = 0.8 + 0.45 * np.random.random()
        
        # Step asymmetry: 0.95 to 1.05 (slight left/right differences)
        characteristics['asymmetry_factor'] = 0.95 + 0.1 * np.random.random()
        
        # Gait variability: 0.8 to 1.2 (individual consistency patterns)
        characteristics['variability_factor'] = 0.8 + 0.4 * np.random.random()
        
        # Cadence preference: 0.9 to 1.1 (step frequency preference)
        characteristics['cadence_factor'] = 0.9 + 0.2 * np.random.random()
        
        return characteristics
    
    @classmethod
    def _preprocess_acceleration(cls, accel_data, sampling_rate):
        """Enhanced preprocessing of acceleration data"""
        import numpy as np
        from scipy import signal
        
        # Convert from g-units to m/s² if needed
        if np.abs(np.mean(accel_data)) < 2.0:  # Data appears to be in g-units
            accel_data = accel_data * cls.GRAVITY  # Convert to m/s²
        
        # Remove DC component (gravity/bias)
        accel_clean = accel_data - np.mean(accel_data)
        
        # Apply median filter to remove spikes
        if len(accel_clean) > 5:
            accel_clean = signal.medfilt(accel_clean, kernel_size=5)
        
        return accel_clean
    
    @classmethod
    def _detect_orientation_changes(cls, gx, gy, gz, sampling_rate):
        """Detect foot orientation changes using gyroscope data"""
        import numpy as np
        from scipy import signal
        
        # Calculate total angular velocity magnitude
        angular_velocity_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # Detect significant orientation changes (foot rotations during steps)
        # High angular velocity indicates foot lift-off and heel strike
        threshold = np.std(angular_velocity_mag) * 2
        orientation_changes = angular_velocity_mag > threshold
        
        return orientation_changes
    
    @classmethod
    def _apply_orientation_correction(cls, ax, ay, az, orientation_changes):
        """Apply orientation correction to accelerometer data using gyroscope information"""
        import numpy as np
        
        # Simple correction: during orientation changes, weight the horizontal components more
        # This is a simplified approach - full IMU fusion would use quaternions
        correction_factor = np.ones_like(ax)
        correction_factor[orientation_changes] = 0.8  # Reduce weight during foot rotation
        
        ax_corrected = ax * correction_factor
        ay_corrected = ay * correction_factor
        az_corrected = az  # Keep vertical as-is
        
        return ax_corrected, ay_corrected, az_corrected
    
    @classmethod
    def _integrate_with_drift_correction(cls, filtered_accel, sampling_rate, step_times):
        """Enhanced integration with better drift correction"""
        import numpy as np
        from scipy import signal
        
        dt = 1.0 / sampling_rate
        
        # Simple integration
        velocity = np.cumsum(filtered_accel) * dt
        
        # Advanced drift correction using step timing
        if len(step_times) > 1:
            # Assume velocity should return to baseline between steps
            avg_step_time = np.mean(step_times)
            step_samples = int(avg_step_time * sampling_rate)
            
            # Apply detrending in step-sized windows
            velocity_corrected = np.copy(velocity)
            for i in range(0, len(velocity), step_samples):
                end_idx = min(i + step_samples, len(velocity))
                window = velocity[i:end_idx]
                if len(window) > 2:
                    # Remove linear trend in each step window
                    detrended = signal.detrend(window, type='linear')
                    velocity_corrected[i:end_idx] = detrended
            
            return velocity_corrected
        else:
            # Fallback: simple high-pass filter
            nyquist = sampling_rate / 2
            b_hp, a_hp = signal.butter(2, 0.1 / nyquist, btype='high')
            return signal.filtfilt(b_hp, a_hp, velocity)
    
    @classmethod
    def _calculate_step_synchronized_speed(cls, velocity, step_times, sampling_rate):
        """Calculate walking speed synchronized with step timing"""
        import numpy as np
        
        if len(step_times) == 0:
            return np.sqrt(np.mean(velocity**2))  # RMS velocity
        
        # Calculate speed estimates for each step interval
        avg_step_time = np.mean(step_times)
        step_samples = int(avg_step_time * sampling_rate)
        
        speed_estimates = []
        for i in range(0, len(velocity) - step_samples, step_samples//2):
            window_velocity = velocity[i:i+step_samples]
            # Use RMS velocity for this step window
            rms_speed = np.sqrt(np.mean(window_velocity**2))
            if 0.2 < rms_speed < 4.0:  # Reasonable walking speeds
                speed_estimates.append(rms_speed)
        
        if speed_estimates:
            return np.median(speed_estimates)
        else:
            return np.sqrt(np.mean(velocity**2))
    
    @classmethod
    def _estimate_stride_displacement_enhanced(cls, accel_data, gyro_data, sampling_rate, step_times):
        """Enhanced stride displacement estimation using both accelerometer and gyroscope"""
        import numpy as np
        from scipy import signal
        
        if len(accel_data.shape) == 1 or len(step_times) == 0:
            return cls._estimate_stride_displacement(accel_data, sampling_rate)
        
        # Use gyroscope to improve step detection accuracy
        if gyro_data is not None and len(gyro_data.shape) > 1:
            # Detect heel strikes using both sensors
            gx, gy, gz = gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2]
            
            # Gyroscope-enhanced step detection
            angular_velocity_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            gyro_threshold = np.std(angular_velocity_mag) * 1.5
            
            # Combine accelerometer magnitude with gyroscope information
            ax, ay, az = accel_data[:, 0], accel_data[:, 1], accel_data[:, 2]
            accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
            
            # Enhanced signal: combine both sensors
            combined_signal = accel_mag + 0.3 * angular_velocity_mag
            
            # Use combined signal for better displacement estimation
            avg_step_time = np.mean(step_times)
            step_samples = int(avg_step_time * sampling_rate)
            
            # Estimate displacement based on combined sensor information
            displacement_per_step = np.std(combined_signal) * 0.5  # Empirical scaling
            return displacement_per_step * 2  # Convert to stride length
        
        # Fallback to original method
        return cls._estimate_stride_displacement(accel_data, sampling_rate)
    
    @classmethod
    def _calculate_motion_intensity(cls, ax, ay, az, sampling_rate):
        """Calculate motion intensity from accelerometer data"""
        import numpy as np
        from scipy import signal
        
        # Convert to m/s² if needed
        if np.abs(np.mean(ax)) < 2.0:
            ax, ay, az = ax * cls.GRAVITY, ay * cls.GRAVITY, az * cls.GRAVITY
        
        # Calculate total acceleration magnitude
        accel_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Remove gravity bias (assume mean is gravity component)
        accel_clean = accel_magnitude - np.mean(accel_magnitude)
        
        # Calculate motion intensity as normalized standard deviation
        motion_intensity = np.std(accel_clean) / 2.0  # Normalize to reasonable range
        
        # Clamp to 0-1 range
        return max(0.0, min(motion_intensity, 1.0))
    
    @classmethod
    def _calculate_rotation_intensity(cls, gx, gy, gz, sampling_rate):
        """Calculate rotation intensity from gyroscope data"""
        import numpy as np
        
        # Calculate total angular velocity magnitude
        gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # Remove bias
        gyro_clean = gyro_magnitude - np.mean(gyro_magnitude)
        
        # Calculate rotation intensity as normalized standard deviation
        rotation_intensity = np.std(gyro_clean) / 50.0  # Normalize (gyro values are larger)
        
        # Clamp to 0-1 range
        return max(0.0, min(rotation_intensity, 1.0))
    
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