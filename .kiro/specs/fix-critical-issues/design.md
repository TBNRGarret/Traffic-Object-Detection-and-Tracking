# Technical Design Document

## Introduction

This document provides the technical design for fixing critical issues in the Traffic Object Detection and Tracking system. The system is a YOLOv11-based application that requires refactoring to eliminate hardcoded paths, add comprehensive error handling, improve documentation, and enable Docker deployment with model serving capabilities.

## Architecture Overview

The refactored system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Training     │  │ Tracking     │  │ Utilities    │      │
│  │ Scripts      │  │ Scripts      │  │ Scripts      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌─────────────────────────────┼─────────────────────────────────┐
│                    Infrastructure Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Config       │  │ Error        │  │ Logging      │       │
│  │ Manager      │  │ Handler      │  │ System       │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────────────────────────────────────────────┘
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼──────────────┐
│                      Storage Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Models       │  │ Datasets     │  │ Videos       │       │
│  │ (.pt, .onnx) │  │ (images)     │  │ (.mp4)       │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Configuration Externalization**: All paths and configurable parameters are loaded from environment variables
2. **Fail-Fast Validation**: Path and configuration validation occurs at startup before any processing
3. **Graceful Degradation**: Non-critical errors are logged and processing continues when possible
4. **Comprehensive Logging**: All operations, errors, and state changes are logged with appropriate levels
5. **Docker-First Deployment**: The system is designed to run in containerized environments with HTTP model serving

## Component Design

### 1. Configuration Management Module

**File**: `config.py`

**Purpose**: Centralized configuration loading and validation from environment variables.

**Class**: `ConfigManager`

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

@dataclass
class PathConfig:
    """Cấu hình đường dẫn cho hệ thống / Path configuration for the system"""
    data_root: Path
    model_path: Path
    video_source: Path
    output_dir: Path
    dataset_path: Path
    log_dir: Path

class ConfigManager:
    """Quản lý cấu hình từ biến môi trường / Manages configuration from environment variables"""
    
    # Giá trị mặc định / Default values
    DEFAULTS = {
        'DATA_ROOT': './data',
        'MODEL_PATH': './models',
        'VIDEO_SOURCE': './videos',
        'OUTPUT_DIR': './output',
        'DATASET_PATH': './dataset',
        'LOG_DIR': './logs'
    }
    
    def __init__(self, env_file: str = '.env'):
        """
        Khởi tạo config manager / Initialize config manager
        
        Args:
            env_file: Đường dẫn đến file .env / Path to .env file
        """
        load_dotenv(env_file)
        self._config: Optional[PathConfig] = None
    
    def load_config(self) -> PathConfig:
        """
        Tải cấu hình từ biến môi trường / Load configuration from environment variables
        
        Returns:
            PathConfig: Đối tượng cấu hình đường dẫn / Path configuration object
        """
        config = PathConfig(
            data_root=Path(os.getenv('DATA_ROOT', self.DEFAULTS['DATA_ROOT'])),
            model_path=Path(os.getenv('MODEL_PATH', self.DEFAULTS['MODEL_PATH'])),
            video_source=Path(os.getenv('VIDEO_SOURCE', self.DEFAULTS['VIDEO_SOURCE'])),
            output_dir=Path(os.getenv('OUTPUT_DIR', self.DEFAULTS['OUTPUT_DIR'])),
            dataset_path=Path(os.getenv('DATASET_PATH', self.DEFAULTS['DATASET_PATH'])),
            log_dir=Path(os.getenv('LOG_DIR', self.DEFAULTS['LOG_DIR']))
        )
        self._config = config
        return config
    
    def validate_paths(self, config: PathConfig) -> list[str]:
        """
        Kiểm tra tính hợp lệ của đường dẫn / Validate path configuration
        
        Args:
            config: Cấu hình cần kiểm tra / Configuration to validate
            
        Returns:
            list[str]: Danh sách lỗi (rỗng nếu hợp lệ) / List of errors (empty if valid)
        """
        errors = []
        
        # Kiểm tra quyền đọc cho đường dẫn đầu vào / Check read permissions for input paths
        input_paths = [config.dataset_path, config.video_source]
        for path in input_paths:
            if path.exists() and not os.access(path, os.R_OK):
                errors.append(f"No read permission for input path: {path}")
        
        # Kiểm tra quyền ghi cho đường dẫn đầu ra / Check write permissions for output paths
        output_paths = [config.output_dir, config.log_dir]
        for path in output_paths:
            if path.exists() and not os.access(path, os.W_OK):
                errors.append(f"No write permission for output path: {path}")
            elif not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create output directory {path}: {e}")
        
        return errors
    
    @property
    def config(self) -> PathConfig:
        """Lấy cấu hình hiện tại / Get current configuration"""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
```

**Key Features**:
- Loads configuration from `.env` file using `python-dotenv`
- Provides sensible defaults for all paths (relative to project root)
- Validates path existence and permissions
- Creates output directories if they don't exist
- Returns detailed error messages for invalid configurations



### 2. Logging Infrastructure Module

**File**: `logger.py`

**Purpose**: Centralized logging configuration with console and file output.

**Class**: `SystemLogger`

```python
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class SystemLogger:
    """Hệ thống logging tập trung / Centralized logging system"""
    
    def __init__(self, name: str, log_dir: Path, level: int = logging.INFO):
        """
        Khởi tạo logger / Initialize logger
        
        Args:
            name: Tên logger / Logger name
            log_dir: Thư mục chứa file log / Directory for log files
            level: Mức độ logging / Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Xóa handlers cũ / Clear old handlers
        
        # Tạo thư mục log nếu chưa tồn tại / Create log directory if not exists
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Định dạng log / Log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler với timestamp / File handler with timestamp
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Ghi log INFO / Log INFO message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Ghi log WARNING / Log WARNING message"""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """
        Ghi log ERROR / Log ERROR message
        
        Args:
            message: Thông điệp lỗi / Error message
            exc_info: Có ghi stack trace không / Whether to include stack trace
        """
        self.logger.error(message, exc_info=exc_info)
    
    def exception(self, message: str):
        """Ghi log exception với stack trace / Log exception with stack trace"""
        self.logger.exception(message)
```

**Key Features**:
- Dual output to console and timestamped log files
- Configurable log levels (INFO, WARNING, ERROR)
- Automatic stack trace logging for exceptions
- UTF-8 encoding support for Vietnamese text
- Timestamp in all log messages

### 3. Error Handling Module

**File**: `error_handler.py`

**Purpose**: Centralized error handling with descriptive messages and graceful degradation.

**Classes**: `PathError`, `ModelError`, `VideoError`, `ErrorHandler`

```python
from pathlib import Path
from typing import Optional, Callable, Any
from logger import SystemLogger

class PathError(Exception):
    """Lỗi liên quan đến đường dẫn / Path-related error"""
    def __init__(self, path: Path, message: str):
        self.path = path
        self.message = message
        super().__init__(f"Path error for '{path}': {message}")

class ModelError(Exception):
    """Lỗi liên quan đến model / Model-related error"""
    def __init__(self, model_path: Path, message: str):
        self.model_path = model_path
        self.message = message
        super().__init__(f"Model error for '{model_path}': {message}")

class VideoError(Exception):
    """Lỗi liên quan đến video / Video-related error"""
    def __init__(self, video_path: Path, message: str):
        self.video_path = video_path
        self.message = message
        super().__init__(f"Video error for '{video_path}': {message}")

class ErrorHandler:
    """Xử lý lỗi tập trung / Centralized error handler"""
    
    def __init__(self, logger: SystemLogger):
        """
        Khởi tạo error handler / Initialize error handler
        
        Args:
            logger: Logger để ghi lỗi / Logger for error recording
        """
        self.logger = logger
        self.error_count = 0
        self.success_count = 0
    
    def validate_file_exists(self, file_path: Path, file_type: str = "file") -> None:
        """
        Kiểm tra file tồn tại / Validate file exists
        
        Args:
            file_path: Đường dẫn file / File path
            file_type: Loại file / File type description
            
        Raises:
            PathError: Nếu file không tồn tại / If file does not exist
        """
        if not file_path.exists():
            raise PathError(
                file_path,
                f"{file_type} does not exist. Please check the path and try again."
            )
        if not file_path.is_file():
            raise PathError(
                file_path,
                f"Path exists but is not a file. Expected a {file_type}."
            )
    
    def validate_directory_exists(self, dir_path: Path, create: bool = False) -> None:
        """
        Kiểm tra thư mục tồn tại / Validate directory exists
        
        Args:
            dir_path: Đường dẫn thư mục / Directory path
            create: Tạo thư mục nếu chưa tồn tại / Create if not exists
            
        Raises:
            PathError: Nếu thư mục không tồn tại và create=False / If directory does not exist and create=False
        """
        if not dir_path.exists():
            if create:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created directory: {dir_path}")
                except Exception as e:
                    raise PathError(dir_path, f"Cannot create directory: {e}")
            else:
                raise PathError(
                    dir_path,
                    "Directory does not exist. Set create=True to create it automatically."
                )
    
    def handle_video_error(self, video_path: Path, error: Exception, continue_processing: bool = True) -> bool:
        """
        Xử lý lỗi video / Handle video processing error
        
        Args:
            video_path: Đường dẫn video / Video path
            error: Exception xảy ra / Exception that occurred
            continue_processing: Tiếp tục xử lý hay không / Whether to continue processing
            
        Returns:
            bool: True nếu nên tiếp tục / True if should continue processing
        """
        self.error_count += 1
        self.logger.error(
            f"Error processing video '{video_path}': {error}",
            exc_info=True
        )
        return continue_processing
    
    def handle_model_error(self, model_path: Path, error: Exception) -> None:
        """
        Xử lý lỗi model (critical) / Handle model error (critical)
        
        Args:
            model_path: Đường dẫn model / Model path
            error: Exception xảy ra / Exception that occurred
            
        Raises:
            ModelError: Luôn raise vì đây là lỗi critical / Always raises as this is critical
        """
        self.logger.error(
            f"Critical model error for '{model_path}': {error}",
            exc_info=True
        )
        raise ModelError(
            model_path,
            f"Cannot load model. Please ensure the model file exists at: {model_path}"
        )
    
    def safe_execute(
        self,
        func: Callable,
        *args,
        error_message: str = "Operation failed",
        critical: bool = False,
        **kwargs
    ) -> Optional[Any]:
        """
        Thực thi hàm với xử lý lỗi / Execute function with error handling
        
        Args:
            func: Hàm cần thực thi / Function to execute
            *args: Tham số cho hàm / Function arguments
            error_message: Thông điệp lỗi / Error message
            critical: Lỗi critical hay không / Whether error is critical
            **kwargs: Tham số keyword cho hàm / Function keyword arguments
            
        Returns:
            Kết quả của hàm hoặc None nếu lỗi / Function result or None if error
        """
        try:
            result = func(*args, **kwargs)
            self.success_count += 1
            return result
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"{error_message}: {e}", exc_info=True)
            if critical:
                raise
            return None
    
    def get_summary(self) -> dict[str, int]:
        """
        Lấy tóm tắt kết quả xử lý / Get processing summary
        
        Returns:
            dict: Số lượng thành công và thất bại / Success and failure counts
        """
        return {
            'success': self.success_count,
            'failed': self.error_count,
            'total': self.success_count + self.error_count
        }
```

**Key Features**:
- Custom exception types for different error categories
- Path validation with descriptive error messages
- Graceful degradation for non-critical errors (video processing)
- Critical error handling for model loading
- Operation counting for summary reporting
- Safe execution wrapper for automatic error handling



### 4. Refactored Training Scripts

**Files**: `yolo.py`, `vnyolo.py`

**Purpose**: Model training with environment-based configuration and error handling.

**Example Implementation** (`yolo.py`):

```python
from ultralytics import YOLO
from pathlib import Path
import sys

from config import ConfigManager
from logger import SystemLogger
from error_handler import ErrorHandler

def main():
    # Khởi tạo cấu hình / Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Khởi tạo logger / Initialize logger
    logger = SystemLogger('yolo_training', config.log_dir)
    logger.info("Starting YOLO training script")
    
    # Khởi tạo error handler / Initialize error handler
    error_handler = ErrorHandler(logger)
    
    # Kiểm tra cấu hình / Validate configuration
    errors = config_manager.validate_paths(config)
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Đường dẫn data.yaml / Path to data.yaml
    data_yaml = config.data_root / "data.yaml"
    
    try:
        # Kiểm tra file tồn tại / Validate file exists
        error_handler.validate_file_exists(data_yaml, "data.yaml")
        
        logger.info(f"Loading YOLO model from: yolo11n.pt")
        model = YOLO("yolo11n.pt")
        
        logger.info(f"Starting training with data: {data_yaml}")
        model.train(
            data=str(data_yaml),
            epochs=20,
            imgsz=640,
            batch=16,
            device=0,
            
            # Tăng cường dữ liệu cho giao thông / Traffic-aware augmentation
            flipud=0.0,  # Không lật dọc / No vertical flip
            fliplr=0.5,  # Lật ngang 50% / Horizontal flip 50%
            degrees=0.1,  # Xoay nhẹ / Slight rotation
            mosaic=1.0,  # Mosaic augmentation
            
            workers=4  # Số worker tải dữ liệu / Number of data loading workers
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        error_handler.handle_model_error(Path("yolo11n.pt"), e)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Key Changes**:
- Removed hardcoded absolute paths
- Added configuration loading from environment variables
- Added comprehensive logging
- Added path validation before processing
- Added error handling with descriptive messages
- Added bilingual comments

### 5. Refactored Tracking Script

**File**: `tracking_yolo.py`

**Purpose**: Object tracking with environment-based configuration and graceful degradation.

```python
from ultralytics import YOLO
from pathlib import Path
import sys

from config import ConfigManager
from logger import SystemLogger
from error_handler import ErrorHandler

def main():
    # Khởi tạo cấu hình / Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Khởi tạo logger / Initialize logger
    logger = SystemLogger('yolo_tracking', config.log_dir)
    logger.info("Starting YOLO tracking script")
    
    # Khởi tạo error handler / Initialize error handler
    error_handler = ErrorHandler(logger)
    
    # Kiểm tra cấu hình / Validate configuration
    errors = config_manager.validate_paths(config)
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Đường dẫn model / Model path
    model_path = config.model_path / "best.pt"
    
    try:
        # Kiểm tra model tồn tại / Validate model exists
        error_handler.validate_file_exists(model_path, "model file")
        
        logger.info(f"Loading trained model from: {model_path}")
        model = YOLO(str(model_path))
        
        # Lấy danh sách video / Get video list
        video_source = Path(config.video_source)
        if video_source.is_file():
            video_files = [video_source]
        elif video_source.is_dir():
            video_files = list(video_source.glob("*.mp4"))
        else:
            raise PathError(video_source, "Video source is neither a file nor a directory")
        
        logger.info(f"Found {len(video_files)} video(s) to process")
        
        # Xử lý từng video / Process each video
        for video_path in video_files:
            logger.info(f"Processing video: {video_path.name}")
            
            try:
                # Chạy tracking / Run tracking
                for _ in model.track(
                    source=str(video_path),
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    iou=0.5,
                    imgsz=640,
                    device=0,
                    save=True,
                    stream=True,
                    show_labels=True,
                    show_conf=False,
                    project=str(config.output_dir),
                    name=f"track_{video_path.stem}"
                ):
                    pass
                
                error_handler.success_count += 1
                logger.info(f"Successfully processed: {video_path.name}")
                
            except Exception as e:
                # Xử lý lỗi video và tiếp tục / Handle video error and continue
                error_handler.handle_video_error(video_path, e, continue_processing=True)
                logger.warning(f"Skipping video due to error: {video_path.name}")
        
        # In tóm tắt / Print summary
        summary = error_handler.get_summary()
        logger.info(f"Processing complete: {summary['success']} succeeded, {summary['failed']} failed")
        
    except Exception as e:
        error_handler.handle_model_error(model_path, e)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Key Features**:
- Processes multiple videos from a directory
- Continues processing if one video fails
- Reports summary of successful and failed operations
- Saves output to configured output directory
- Comprehensive logging of all operations

### 6. Refactored Utility Scripts

**File**: `framecutter.py`

**Purpose**: Extract frames from videos with error handling and graceful degradation.

```python
import cv2
from pathlib import Path
import sys

from config import ConfigManager
from logger import SystemLogger
from error_handler import ErrorHandler

def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_frames: int,
    logger: SystemLogger,
    error_handler: ErrorHandler
) -> bool:
    """
    Trích xuất khung hình từ video / Extract frames from video
    
    Args:
        video_path: Đường dẫn video / Video path
        output_dir: Thư mục đầu ra / Output directory
        target_frames: Số khung hình mục tiêu / Target frame count
        logger: Logger
        error_handler: Error handler
        
    Returns:
        bool: True nếu thành công / True if successful
    """
    try:
        # Tạo thư mục đầu ra / Create output directory
        video_output_dir = output_dir / video_path.stem
        error_handler.validate_directory_exists(video_output_dir, create=True)
        
        # Mở video / Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoError(video_path, "Cannot open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total_frames // target_frames, 1)
        
        logger.info(f"Processing {video_path.name}: {total_frames} frames, extracting every {interval}th frame")
        
        frame_id = 0
        saved = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or saved >= target_frames:
                break
            
            if frame_id % interval == 0:
                out_path = video_output_dir / f"frame_{saved:05d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
            
            frame_id += 1
        
        cap.release()
        logger.info(f"{video_path.stem}: saved {saved} frames")
        return True
        
    except Exception as e:
        error_handler.handle_video_error(video_path, e, continue_processing=True)
        return False

def main():
    # Khởi tạo cấu hình / Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Khởi tạo logger / Initialize logger
    logger = SystemLogger('framecutter', config.log_dir)
    logger.info("Starting frame extraction script")
    
    # Khởi tạo error handler / Initialize error handler
    error_handler = ErrorHandler(logger)
    
    # Kiểm tra cấu hình / Validate configuration
    errors = config_manager.validate_paths(config)
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Thư mục video và đầu ra / Video and output directories
    video_dir = Path(config.video_source)
    output_root = config.output_dir / "frames"
    
    try:
        error_handler.validate_directory_exists(video_dir)
        error_handler.validate_directory_exists(output_root, create=True)
        
        # Lấy danh sách video / Get video list
        video_files = list(video_dir.glob("*.mp4"))
        logger.info(f"Found {len(video_files)} video(s) to process")
        
        # Xử lý từng video / Process each video
        for video_path in video_files:
            # Trường hợp đặc biệt: Nguyen Trai rush hour / Special case: Nguyen Trai rush hour
            if video_path.stem.lower() == "nguyentraihn":
                target_frames = 250
            else:
                target_frames = 200
            
            success = extract_frames(
                video_path,
                output_root,
                target_frames,
                logger,
                error_handler
            )
            
            if success:
                error_handler.success_count += 1
            else:
                error_handler.error_count += 1
        
        # In tóm tắt / Print summary
        summary = error_handler.get_summary()
        logger.info(f"Extraction complete: {summary['success']} succeeded, {summary['failed']} failed")
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Key Features**:
- Processes multiple videos with graceful degradation
- Creates output directories automatically
- Validates video files before processing
- Reports detailed progress and summary
- Handles special cases (different frame counts per video)



### 7. Model Export Script

**File**: `export_onnx.py`

**Purpose**: Export trained models to ONNX format with error handling.

```python
from ultralytics import YOLO
from pathlib import Path
import sys

from config import ConfigManager
from logger import SystemLogger
from error_handler import ErrorHandler

def main():
    # Khởi tạo cấu hình / Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Khởi tạo logger / Initialize logger
    logger = SystemLogger('onnx_export', config.log_dir)
    logger.info("Starting ONNX export script")
    
    # Khởi tạo error handler / Initialize error handler
    error_handler = ErrorHandler(logger)
    
    # Kiểm tra cấu hình / Validate configuration
    errors = config_manager.validate_paths(config)
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Đường dẫn model / Model path
    model_path = config.model_path / "best.pt"
    
    try:
        # Kiểm tra model tồn tại / Validate model exists
        error_handler.validate_file_exists(model_path, "model file")
        
        logger.info(f"Loading model from: {model_path}")
        model = YOLO(str(model_path))
        
        logger.info("Exporting model to ONNX format")
        model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=True
        )
        
        # Đường dẫn ONNX đầu ra / Output ONNX path
        onnx_path = model_path.with_suffix('.onnx')
        logger.info(f"Model exported successfully to: {onnx_path}")
        
    except Exception as e:
        error_handler.handle_model_error(model_path, e)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 8. Docker Configuration

**File**: `Dockerfile`

**Purpose**: Containerized deployment with model serving capability.

```dockerfile
# Cú pháp Dockerfile / Dockerfile syntax
# syntax=docker/dockerfile:1

# Phiên bản Python / Python version
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Ngăn Python ghi file pyc / Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ngăn Python buffer stdout/stderr / Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Thư mục làm việc / Working directory
WORKDIR /app

# Cài đặt dependencies hệ thống cho OpenCV / Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tạo user không có quyền root / Create non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Cài đặt dependencies Python / Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn / Copy source code
COPY . .

# Tạo thư mục cần thiết / Create necessary directories
RUN mkdir -p /app/models /app/output /app/logs && \
    chown -R appuser:appuser /app

# Chuyển sang user không có quyền root / Switch to non-privileged user
USER appuser

# Expose port cho HTTP serving / Expose port for HTTP serving
EXPOSE 8000

# Biến môi trường mặc định / Default environment variables
ENV MODEL_PATH=/app/models
ENV OUTPUT_DIR=/app/output
ENV LOG_DIR=/app/logs

# Chạy ứng dụng / Run application
CMD ["python", "serve_model.py"]
```

**Key Features**:
- Based on Python 3.11 slim image
- Installs system dependencies for OpenCV
- Installs Python dependencies from requirements.txt
- Creates necessary directories with proper permissions
- Runs as non-root user for security
- Exposes port 8000 for HTTP serving
- Sets default environment variables

### 9. Model Serving Script

**File**: `serve_model.py`

**Purpose**: HTTP API for model inference in Docker container.

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import sys
from typing import List, Dict, Any

from config import ConfigManager
from logger import SystemLogger
from error_handler import ErrorHandler, ModelError

# Khởi tạo FastAPI app / Initialize FastAPI app
app = FastAPI(title="YOLO Traffic Detection API")

# Biến toàn cục / Global variables
model = None
logger = None
error_handler = None

@app.on_event("startup")
async def startup_event():
    """Khởi động ứng dụng và tải model / Startup application and load model"""
    global model, logger, error_handler
    
    # Khởi tạo cấu hình / Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Khởi tạo logger / Initialize logger
    logger = SystemLogger('model_serving', config.log_dir)
    logger.info("Starting model serving API")
    
    # Khởi tạo error handler / Initialize error handler
    error_handler = ErrorHandler(logger)
    
    # Đường dẫn model / Model path
    model_path = config.model_path / "best.pt"
    
    try:
        # Kiểm tra model tồn tại / Validate model exists
        error_handler.validate_file_exists(model_path, "model file")
        
        logger.info(f"Loading model from: {model_path}")
        model = YOLO(str(model_path))
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

@app.get("/health")
async def health_check():
    """Kiểm tra sức khỏe API / Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Dự đoán đối tượng trong ảnh / Predict objects in image
    
    Args:
        file: File ảnh upload / Uploaded image file
        
    Returns:
        JSONResponse: Kết quả dự đoán / Prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Đọc ảnh / Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Chạy dự đoán / Run prediction
        results = model(img)
        
        # Chuyển đổi kết quả sang JSON / Convert results to JSON
        predictions: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                predictions.append({
                    "class": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })
        
        logger.info(f"Processed image: {file.filename}, found {len(predictions)} objects")
        
        return JSONResponse(content={
            "filename": file.filename,
            "predictions": predictions,
            "count": len(predictions)
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Chạy server / Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Key Features**:
- FastAPI-based HTTP API
- Health check endpoint
- Image upload and prediction endpoint
- JSON response format with bounding boxes and class names
- Comprehensive error handling
- Logging of all requests
- Graceful startup failure if model is missing

### 10. Configuration Files

**File**: `.env.example`

**Purpose**: Template for environment variable configuration.

```bash
# Cấu hình đường dẫn cho hệ thống Traffic Object Detection / Path configuration for Traffic Object Detection system

# Thư mục gốc chứa dữ liệu / Root directory containing data
# Mặc định: ./data / Default: ./data
DATA_ROOT=./data

# Đường dẫn đến model đã train / Path to trained models
# Mặc định: ./models / Default: ./models
# Ví dụ: ./runs/detect/train4/weights hoặc /app/models trong Docker / Example: ./runs/detect/train4/weights or /app/models in Docker
MODEL_PATH=./models

# Đường dẫn đến video nguồn / Path to source videos
# Mặc định: ./videos / Default: ./videos
# Có thể là file hoặc thư mục / Can be a file or directory
VIDEO_SOURCE=./videos

# Thư mục đầu ra cho kết quả / Output directory for results
# Mặc định: ./output / Default: ./output
OUTPUT_DIR=./output

# Đường dẫn đến dataset / Path to dataset
# Mặc định: ./dataset / Default: ./dataset
DATASET_PATH=./dataset

# Thư mục chứa log files / Directory for log files
# Mặc định: ./logs / Default: ./logs
LOG_DIR=./logs
```

**File**: `requirements.txt`

**Purpose**: Python dependency specification.

```txt
# Core dependencies / Dependencies chính
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0

# ONNX support / Hỗ trợ ONNX
onnx>=1.14.0
onnxruntime>=1.15.0

# Configuration management / Quản lý cấu hình
python-dotenv>=1.0.0

# HTTP serving / Phục vụ HTTP
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6

# Utilities / Tiện ích
numpy>=1.24.0
Pillow>=10.0.0
pyyaml>=6.0
```

**File**: `data.yaml` (updated)

**Purpose**: Dataset configuration with relative paths.

```yaml
# Đường dẫn tương đối đến dataset / Relative paths to dataset
train: ./dataset/uadetrac/train/images
val: ./dataset/uadetrac/valid/images
test: ./dataset/uadetrac/test/images

# Số lượng class / Number of classes
nc: 4

# Tên các class / Class names
names: ['bus', 'car', 'truck', 'van']

# Thông tin Roboflow / Roboflow information
roboflow:
  workspace: rjacaac1
  project: ua-detrac-dataset-10k
  version: 1
  license: Private
  url: https://universe.roboflow.com/rjacaac1/ua-detrac-dataset-10k/dataset/1
```

## Data Models

### PathConfig

```python
@dataclass
class PathConfig:
    """Cấu hình đường dẫn / Path configuration"""
    data_root: Path
    model_path: Path
    video_source: Path
    output_dir: Path
    dataset_path: Path
    log_dir: Path
```

### ProcessingSummary

```python
@dataclass
class ProcessingSummary:
    """Tóm tắt kết quả xử lý / Processing summary"""
    success_count: int
    error_count: int
    total_count: int
    errors: List[str]
```

### PredictionResult

```python
@dataclass
class PredictionResult:
    """Kết quả dự đoán / Prediction result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
```

## Error Handling Strategy

### Error Categories

1. **Critical Errors** (terminate immediately):
   - Model file not found
   - Invalid configuration preventing startup
   - Missing required dependencies

2. **Non-Critical Errors** (log and continue):
   - Individual video processing failures
   - Individual frame extraction failures
   - Network errors during inference (retry)

### Error Flow

```
┌─────────────────┐
│  Operation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Try Execute    │
└────────┬────────┘
         │
         ▼
    ┌────────┐
    │Success?│
    └───┬────┘
        │
   ┌────┴────┐
   │         │
  Yes       No
   │         │
   │         ▼
   │    ┌─────────────┐
   │    │  Critical?  │
   │    └──────┬──────┘
   │           │
   │      ┌────┴────┐
   │      │         │
   │     Yes       No
   │      │         │
   │      ▼         ▼
   │  ┌──────┐  ┌──────────┐
   │  │ Exit │  │Log & Skip│
   │  └──────┘  └─────┬────┘
   │                  │
   └──────────────────┘
            │
            ▼
    ┌──────────────┐
    │   Continue   │
    └──────────────┘
```



## Deployment Architecture

### Local Development

```
┌─────────────────────────────────────────────────────────┐
│                   Developer Machine                      │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  .env      │  │  Python    │  │  CUDA      │       │
│  │  Config    │  │  3.11+     │  │  (GPU)     │       │
│  └────────────┘  └────────────┘  └────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────┐      │
│  │           Application Scripts                 │      │
│  │  • yolo.py (training)                        │      │
│  │  • tracking_yolo.py (tracking)               │      │
│  │  • framecutter.py (preprocessing)            │      │
│  │  • export_onnx.py (model export)             │      │
│  └──────────────────────────────────────────────┘      │
│                                                          │
│  ┌──────────────────────────────────────────────┐      │
│  │        Infrastructure Modules                 │      │
│  │  • config.py (configuration)                 │      │
│  │  • logger.py (logging)                       │      │
│  │  • error_handler.py (error handling)         │      │
│  └──────────────────────────────────────────────┘      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Docker Deployment

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │              FastAPI Application                │    │
│  │                                                 │    │
│  │  ┌──────────────┐      ┌──────────────┐       │    │
│  │  │   /health    │      │   /predict   │       │    │
│  │  │   endpoint   │      │   endpoint   │       │    │
│  │  └──────────────┘      └──────────────┘       │    │
│  │                                                 │    │
│  │  ┌──────────────────────────────────────┐     │    │
│  │  │        YOLO Model (best.pt)          │     │    │
│  │  └──────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Port 8000 ◄──────────────────────────────────────┐    │
│                                                     │    │
└─────────────────────────────────────────────────────────┘
                                                      │
                                                      │
┌─────────────────────────────────────────────────────────┐
│                      Client                              │
│                                                          │
│  POST /predict                                           │
│  Content-Type: multipart/form-data                      │
│  Body: image file                                        │
│                                                          │
│  Response:                                               │
│  {                                                       │
│    "filename": "traffic.jpg",                           │
│    "predictions": [                                      │
│      {                                                   │
│        "class": 1,                                       │
│        "class_name": "car",                             │
│        "confidence": 0.95,                              │
│        "bbox": [100, 200, 300, 400]                     │
│      }                                                   │
│    ],                                                    │
│    "count": 1                                            │
│  }                                                       │
└─────────────────────────────────────────────────────────┘
```

## Testing Strategy

### Unit Tests

Unit tests focus on individual components and specific examples:

1. **Configuration Management**:
   - Test loading valid .env files
   - Test default value fallback
   - Test invalid configuration detection

2. **Path Validation**:
   - Test existing file validation
   - Test missing file error messages
   - Test directory creation

3. **Error Handling**:
   - Test exception raising for missing files
   - Test error message formatting
   - Test critical vs non-critical error classification

4. **Logging**:
   - Test log file creation
   - Test console output
   - Test timestamp formatting

### Integration Tests

Integration tests verify external service interactions:

1. **Docker Build**:
   - Test Dockerfile builds successfully
   - Test container starts without errors
   - Test model loading at startup

2. **Model Serving**:
   - Test HTTP endpoints respond
   - Test inference with sample images
   - Test error handling for invalid inputs

3. **Dependency Installation**:
   - Test pip install from requirements.txt
   - Test all imports work correctly

### Property-Based Tests

Property-based tests verify universal properties across many inputs. See the Correctness Properties section below for detailed property specifications.

## Performance Considerations

### Training Performance

- **Batch Size**: Configurable via training script (default: 16)
- **Workers**: Configurable data loading workers (default: 4)
- **GPU Utilization**: Automatic CUDA device selection (device=0)
- **Augmentation**: Traffic-aware augmentation to improve model robustness

### Inference Performance

- **ONNX Export**: Optimized model format for faster inference
- **Dynamic Batching**: Support for variable input sizes
- **Simplified Model**: ONNX simplification for reduced model size

### Docker Performance

- **Slim Base Image**: Python 3.11 slim reduces image size
- **Layer Caching**: Dependencies installed before code copy for better caching
- **Non-Root User**: Security best practice with minimal overhead

## Security Considerations

### Docker Security

1. **Non-Root User**: Application runs as unprivileged user (UID 10001)
2. **Minimal Base Image**: Python slim image reduces attack surface
3. **No Secrets in Image**: All configuration via environment variables
4. **Read-Only Filesystem**: Application code is read-only after build

### Path Security

1. **Path Validation**: All paths validated before use
2. **Permission Checks**: Read/write permissions verified at startup
3. **No Path Traversal**: Paths resolved to absolute paths to prevent traversal attacks

### API Security

1. **Input Validation**: Image files validated before processing
2. **Error Messages**: No sensitive information in error responses
3. **Rate Limiting**: Can be added via FastAPI middleware (future enhancement)

## Migration Guide

### Step 1: Install Dependencies

```bash
# Cài đặt dependencies / Install dependencies
pip install -r requirements.txt
```

### Step 2: Create Configuration

```bash
# Sao chép file cấu hình mẫu / Copy example configuration
cp .env.example .env

# Chỉnh sửa .env với đường dẫn thực tế / Edit .env with actual paths
nano .env
```

### Step 3: Update Existing Scripts

```bash
# Thêm các module mới / Add new modules
# - config.py
# - logger.py
# - error_handler.py
# - serve_model.py

# Cập nhật các script hiện có / Update existing scripts
# - yolo.py
# - vnyolo.py
# - tracking_yolo.py
# - framecutter.py
# - scriptprelabel.py
# - export_onnx.py
# - onnx_testing.py
# - benchmark_onnx.py
# - chia.py
```

### Step 4: Update data.yaml

```bash
# Thay thế đường dẫn tuyệt đối bằng đường dẫn tương đối / Replace absolute paths with relative paths
# Xem phần "Configuration Files" ở trên / See "Configuration Files" section above
```

### Step 5: Test Locally

```bash
# Chạy training / Run training
python yolo.py

# Chạy tracking / Run tracking
python tracking_yolo.py

# Kiểm tra logs / Check logs
ls -la logs/
```

### Step 6: Build Docker Image

```bash
# Build image / Xây dựng image
docker build -t traffic-detection:latest .

# Chạy container / Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  traffic-detection:latest
```

### Step 7: Test API

```bash
# Kiểm tra health / Check health
curl http://localhost:8000/health

# Gửi ảnh để dự đoán / Send image for prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Environment Variable Retrieval with Fallback

*For any* environment variable name in the system's configuration, when the variable is defined in the environment, the system SHALL retrieve its value; when the variable is not defined, the system SHALL return the documented default value.

**Validates: Requirements 1.3, 1.4**

### Property 2: Hardcoded Path Elimination

*For any* Python source file in the project, the file SHALL NOT contain hardcoded absolute paths (paths starting with drive letters on Windows or root `/` on Unix).

**Validates: Requirements 1.5**

### Property 3: Requirements File Format Validity

*For any* non-comment line in requirements.txt, the line SHALL specify a valid Python package name followed by a valid version specifier (e.g., `>=`, `==`, `~=`).

**Validates: Requirements 2.2, 2.4**

### Property 4: Path Validation and Error Handling

*For any* file path that the system attempts to access, the system SHALL first verify the path exists; if the path does not exist, the system SHALL raise an exception with a descriptive error message including the path and expected file type.

**Validates: Requirements 4.1, 4.2**

### Property 5: Directory Creation or Error Reporting

*For any* required directory path that does not exist, the system SHALL either create the directory with appropriate permissions OR report a descriptive error message indicating the missing path.

**Validates: Requirements 4.3**

### Property 6: Graceful Video Processing Degradation

*For any* batch of video files where some files are invalid or cannot be opened, the system SHALL log the error for each failed video and continue processing the remaining valid videos.

**Validates: Requirements 4.4, 9.1**

### Property 7: File I/O Exception Wrapping

*For any* file I/O operation in the codebase (open, read, write), the operation SHALL be wrapped in a try-except block that catches and handles exceptions.

**Validates: Requirements 4.6**

### Property 8: Error Logging Completeness

*For any* error that occurs during system execution, the logger SHALL record a log entry containing the error type, the file path involved (if applicable), and a timestamp.

**Validates: Requirements 4.7**

### Property 9: Startup Path Validation

*For any* Python script execution, the system SHALL validate all required paths from environment variables before executing the main processing logic.

**Validates: Requirements 5.1**

### Property 10: Invalid Path Reporting and Termination

*For any* configuration containing one or more invalid or inaccessible paths, the system SHALL report ALL invalid paths in the error output and terminate before processing begins.

**Validates: Requirements 5.2**

### Property 11: Permission Validation

*For any* input path, the system SHALL check read permissions; for any output path, the system SHALL check write permissions; if permissions are insufficient, the system SHALL report an actionable error message.

**Validates: Requirements 5.3, 5.4**

### Property 12: Logger Initialization

*For any* Python script in the system, the script SHALL initialize a logger instance at startup before executing any processing logic.

**Validates: Requirements 6.1**

### Property 13: Dual Logging Output

*For any* log message generated by the system, the message SHALL appear in both the console output and a log file.

**Validates: Requirements 6.2**

### Property 14: Appropriate Log Levels

*For any* operation, the system SHALL use INFO level for normal operations, WARNING level for recoverable issues, and ERROR level for failures.

**Validates: Requirements 6.3, 6.4, 6.5**

### Property 15: Exception Stack Trace Logging

*For any* exception that occurs, the logger SHALL record the full stack trace in the log output.

**Validates: Requirements 6.6**

### Property 16: Log Timestamp Presence

*For any* log message, the message SHALL include a timestamp in the format `YYYY-MM-DD HH:MM:SS`.

**Validates: Requirements 6.7**

### Property 17: Bilingual Comment Format

*For any* comment in the codebase, the comment SHALL be in bilingual format with Vietnamese text first, followed by a forward slash separator, followed by English translation (format: `# Vietnamese text / English translation`).

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

### Property 18: Environment Variable Documentation Completeness

*For any* environment variable used by the system, the .env.example file SHALL contain an entry for that variable with an example value, a comment explaining its purpose, documentation of its expected format, and its default value if applicable.

**Validates: Requirements 8.3, 8.4, 8.5**

### Property 19: Frame Processing Graceful Degradation

*For any* batch of frames being processed, if one frame fails to process, the system SHALL log the failure and continue processing the remaining frames.

**Validates: Requirements 9.2**

### Property 20: Non-Critical Error Continuation

*For any* non-critical operation failure, the logger SHALL record the failure and the system SHALL continue execution without terminating.

**Validates: Requirements 9.3**

### Property 21: Error Classification

*For any* error that occurs, the system SHALL correctly classify it as either critical (requiring termination) or non-critical (allowing continuation) based on whether the error prevents core functionality.

**Validates: Requirements 9.4**

### Property 22: Processing Summary Reporting

*For any* processing run that completes (successfully or with errors), the system SHALL report the count of successful operations and the count of failed operations.

**Validates: Requirements 9.5**

### Property 23: JSON Response Format

*For any* valid inference request to the HTTP API, the response SHALL be valid JSON containing a predictions array with class, class_name, confidence, and bbox fields for each detection.

**Validates: Requirements 10.5**

## Conclusion

This design provides a comprehensive solution for fixing critical issues in the Traffic Object Detection and Tracking system. The refactored architecture emphasizes:

1. **Portability**: Environment-based configuration eliminates hardcoded paths
2. **Reliability**: Comprehensive error handling and validation prevent crashes
3. **Observability**: Detailed logging enables debugging and monitoring
4. **Maintainability**: Bilingual comments and clear documentation support international teams
5. **Deployability**: Docker support with HTTP API enables production deployment

The modular design with clear separation between configuration, logging, error handling, and application logic makes the system easier to test, maintain, and extend.

