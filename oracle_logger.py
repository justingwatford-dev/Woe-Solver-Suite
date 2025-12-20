"""
Oracle V4 Logging Utility
Dual logging to file and console
"""
import logging
import sys
from datetime import datetime
import os

class OracleLogger:
    """Handles dual logging for Oracle V4 simulations"""
    
    def __init__(self, storm_name='STORM', storm_year=2005, run_id=None, log_dir='logs'):
        """
        Setup logging to both file and console
        
        Args:
            storm_name: Hurricane name
            storm_year: Hurricane year
            run_id: Optional run identifier
            log_dir: Directory to save logs (default: 'logs')
        """
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_suffix = f"_run{run_id}" if run_id is not None else ""
        self.log_filename = os.path.join(
            log_dir,
            f"oracle_v4_{storm_name}_{storm_year}_{timestamp}{run_suffix}_FULL.log"
        )
        
        # Create logger
        self.logger = logging.getLogger(f'oracle_v4_{timestamp}')
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # File handler (saves everything)
        file_handler = logging.FileHandler(self.log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler (shows on screen)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add both handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log header
        self.info("=" * 80)
        self.info(f"ORACLE V4 SIMULATION LOG")
        self.info("=" * 80)
        self.info(f"Storm: {storm_name} ({storm_year})")
        self.info(f"Log file: {self.log_filename}")
        self.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 80)
        self.info("")
    
    def info(self, message):
        """Log info message to both file and console"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(f"WARNING: {message}")
    
    def error(self, message):
        """Log error message"""
        self.logger.error(f"ERROR: {message}")
    
    def close(self):
        """Close all handlers"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
    
    def get_filename(self):
        """Return the log filename"""
        return self.log_filename


# Global logger instance (set by simulation)
_global_logger = None

def setup_global_logger(storm_name, storm_year, run_id=None, log_dir='logs'):
    """Setup global logger for use throughout codebase"""
    global _global_logger
    _global_logger = OracleLogger(storm_name, storm_year, run_id, log_dir)
    return _global_logger

def get_logger():
    """Get the global logger instance"""
    return _global_logger

def log_info(message):
    """Convenience function to log info"""
    if _global_logger:
        _global_logger.info(message)
    else:
        print(message)

def log_warning(message):
    """Convenience function to log warning"""
    if _global_logger:
        _global_logger.warning(message)
    else:
        print(f"WARNING: {message}")

def log_error(message):
    """Convenience function to log error"""
    if _global_logger:
        _global_logger.error(message)
    else:
        print(f"ERROR: {message}")