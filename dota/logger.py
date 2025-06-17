import logging
import time
import sys
import os
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import threading


class LogLevel(Enum):
    """Enum for log levels with their numeric values"""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


class ColorFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        if hasattr(record, 'use_colors') and getattr(record, 'use_colors', False):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


class Clock:
    """Clock utility for measuring execution time and providing timestamps"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.checkpoints = {}
    
    def reset(self):
        """Reset the clock"""
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.checkpoints.clear()
    
    def elapsed(self) -> float:
        """Get elapsed time since start"""
        return time.time() - self.start_time
    
    def checkpoint(self, name: Optional[str] = None) -> float:
        """Create a checkpoint and return time since last checkpoint"""
        current_time = time.time()
        elapsed_since_last = current_time - self.last_checkpoint
        
        if name:
            self.checkpoints[name] = {
                'time': current_time,
                'elapsed_since_start': current_time - self.start_time,
                'elapsed_since_last': elapsed_since_last
            }
        
        self.last_checkpoint = current_time
        return elapsed_since_last
    
    def get_checkpoint(self, name: str) -> Optional[Dict[str, float]]:
        """Get checkpoint data by name"""
        return self.checkpoints.get(name)
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into human readable format"""
        if seconds < 1:
            return f"{seconds*1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"


class DotaLogger:
    """Complete logger with comprehensive features for the Dota project"""
    
    def __init__(
        self,
        name: str = "DotaLogger",
        level: LogLevel = LogLevel.INFO,
        verbose: bool = False,
        log_file: Optional[str] = None,
        console_output: bool = True,
        use_colors: bool = True,
        include_caller: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        self.name = name
        self.verbose = verbose
        self.use_colors = use_colors and console_output
        self.include_caller = include_caller
        self.clock = Clock()
        self._lock = threading.Lock()
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        self._setup_formatters()
        
        # Setup handlers
        if console_output:
            self._setup_console_handler()
        
        if log_file:
            self._setup_file_handler(log_file, max_file_size, backup_count)
    
    def _setup_formatters(self):
        """Setup formatters for different outputs"""
        # Base format
        base_format = "%(asctime)s | %(levelname)-8s"
        
        if self.include_caller:
            base_format += " | %(name)s:%(funcName)s:%(lineno)d"
        else:
            base_format += " | %(name)s"
        
        base_format += " | %(message)s"
        
        # Console formatter (with colors)
        self.console_formatter = ColorFormatter(
            base_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File formatter (no colors)
        self.file_formatter = logging.Formatter(
            base_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def _setup_console_handler(self):
        """Setup console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: str, max_file_size: int, backup_count: int):
        """Setup rotating file handler"""
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def set_level(self, level: LogLevel):
        """Set logging level"""
        self.logger.setLevel(level.value)
    
    def set_verbose(self, verbose: bool):
        """Enable/disable verbose mode"""
        self.verbose = verbose
    
    def _log(self, level: int, message: str, *args, **kwargs):
        """Internal logging method"""
        with self._lock:
            # Add color flag for console output
            extra = kwargs.get('extra', {})
            extra['use_colors'] = self.use_colors
            kwargs['extra'] = extra
            
            # Add timing information if verbose
            if self.verbose and level >= logging.INFO:
                elapsed = self.clock.elapsed()
                formatted_time = self.clock.format_time(elapsed)
                message = f"[⏱️ {formatted_time}] {message}"
            
            self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, *args, **kwargs)
    
    def warn(self, message: str, *args, **kwargs):
        """Alias for warning"""
        self.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback"""
        kwargs['exc_info'] = True
        self.error(message, *args, **kwargs)
    
    def timer_start(self, operation: Optional[str] = None):
        """Start timing an operation"""
        if operation:
            self.clock.checkpoint(f"start_{operation}")
            if self.verbose:
                self.info(f"⏰ Starting operation: {operation}")
        else:
            self.clock.reset()
            if self.verbose:
                self.info("⏰ Timer started")
        return operation
    
    def timer_end(self, operation: Optional[str] = None):
        """End timing and log the duration"""
        if operation:
            start_checkpoint = self.clock.get_checkpoint(f"start_{operation}")
            if start_checkpoint:
                elapsed = time.time() - start_checkpoint['time']
                formatted_time = self.clock.format_time(elapsed)
                self.info(f"✅ Completed operation '{operation}' in {formatted_time}")
            else:
                self.warning(f"⚠️ No start checkpoint found for operation: {operation}")
        else:
            elapsed = self.clock.elapsed()
            formatted_time = self.clock.format_time(elapsed)
            self.info(f"✅ Operation completed in {formatted_time}")
    
    def checkpoint(self, name: str, message: Optional[str] = None):
        """Create a checkpoint with optional message"""
        elapsed = self.clock.checkpoint(name)
        formatted_time = self.clock.format_time(elapsed)
        
        if message:
            self.info(f"📍 Checkpoint '{name}': {message} (Δt: {formatted_time})")
        else:
            self.info(f"📍 Checkpoint '{name}' (Δt: {formatted_time})")
    
    def progress(self, current: int, total: int, message: str = "Progress"):
        """Log progress information"""
        percentage = (current / total) * 100
        elapsed = self.clock.elapsed()
        
        if current > 0:
            estimated_total = elapsed * total / current
            remaining = estimated_total - elapsed
            eta = self.clock.format_time(remaining)
            self.info(f"📊 {message}: {current}/{total} ({percentage:.1f}%) - ETA: {eta}")
        else:
            self.info(f"📊 {message}: {current}/{total} ({percentage:.1f}%)")
    
    def separator(self, title: Optional[str] = None, char: str = "=", length: int = 80):
        """Log a separator line"""
        if title:
            title_len = len(title)
            if title_len + 4 >= length:
                separator = f"{char * 2} {title} {char * 2}"
            else:
                padding = (length - title_len - 4) // 2
                separator = f"{char * padding} {title} {char * (length - title_len - 4 - padding)}"
        else:
            separator = char * length
        
        self.info(separator)
    
    def system_info(self):
        """Log system information"""
        import platform
        import psutil
        
        self.separator("System Information")
        self.info(f"🖥️  System: {platform.system()} {platform.release()}")
        self.info(f"🐍 Python: {platform.python_version()}")
        self.info(f"💾 Memory: {psutil.virtual_memory().total // (1024**3)} GB")
        self.info(f"🔧 CPU: {psutil.cpu_count()} cores")
        self.info(f"📁 Working Directory: {os.getcwd()}")
        self.separator()
    
    def add_file_handler(self, log_file: str, level: Optional[LogLevel] = None):
        """Add an additional file handler"""
        from logging.handlers import RotatingFileHandler
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.file_formatter)
        
        if level:
            file_handler.setLevel(level.value)
        
        self.logger.addHandler(file_handler)
    
    def get_clock_info(self) -> Dict[str, Any]:
        """Get clock information"""
        return {
            'elapsed_total': self.clock.elapsed(),
            'elapsed_formatted': self.clock.format_time(self.clock.elapsed()),
            'start_time': datetime.fromtimestamp(self.clock.start_time).isoformat(),
            'checkpoints': self.clock.checkpoints
        }


# Convenience function to create a default logger
def get_logger(
    name: str = "DotaLogger",
    level: LogLevel = LogLevel.INFO,
    verbose: bool = False,
    log_file: Optional[str] = None
) -> DotaLogger:
    """Get a configured logger instance"""
    return DotaLogger(
        name=name,
        level=level,
        verbose=verbose,
        log_file=log_file,
        console_output=True,
        use_colors=True
    )