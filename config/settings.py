"""
Configuration management for Face Clustering System
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


class Settings:
    """Configuration manager for the face clustering system"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize settings from configuration file
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self):
        """Validate required configuration sections"""
        required_sections = ['paths', 'face_detection', 'clustering']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'paths.input_folder')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False)
    
    @property
    def input_folder(self) -> str:
        return self.get('paths.input_folder')
    
    @property
    def output_folder(self) -> str:
        return self.get('paths.output_folder')
    
    @property
    def min_face_size(self) -> int:
        return self.get('face_detection.min_face_size', 30)
    
    @property
    def eps(self) -> float:
        return self.get('clustering.eps', 0.45)
    
    @property
    def supported_formats(self) -> List[str]:
        return self.get('image_processing.supported_formats', ['.jpg', '.jpeg', '.png'])
    
    @property
    def debug_enabled(self) -> bool:
        return self.get('debug.enabled', False)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.output_folder,
            self.get('paths.temp_folder'),
            self.get('paths.log_folder'),
            os.path.join(self.output_folder, 'unclustered')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)