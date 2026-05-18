# Implementation Plan: Fix Critical Issues

## Overview

This implementation plan addresses critical issues in the Traffic Object Detection and Tracking system by refactoring the codebase to eliminate hardcoded paths, add comprehensive error handling, improve documentation, and enable Docker deployment with model serving capabilities. The refactoring introduces three core infrastructure modules (configuration management, logging, and error handling) that will be used by all application scripts.

## Tasks

- [ ] 1. Create infrastructure modules
  - [ ] 1.1 Create config.py module with ConfigManager class
    - Implement PathConfig dataclass with all required path fields
    - Implement ConfigManager class with load_config() method
    - Add environment variable loading using python-dotenv
    - Add default values for all configuration paths
    - Add validate_paths() method to check path existence and permissions
    - Create output directories if they don't exist
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.1, 5.2, 5.3, 5.4_

  - [ ] 1.2 Create logger.py module with SystemLogger class
    - Implement SystemLogger class with dual output (console and file)
    - Add support for INFO, WARNING, and ERROR log levels
    - Add timestamp formatting for all log messages
    - Add UTF-8 encoding support for Vietnamese text
    - Create log directory if it doesn't exist
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ] 1.3 Create error_handler.py module with custom exceptions and ErrorHandler class
    - Implement PathError, ModelError, and VideoError exception classes
    - Implement ErrorHandler class with path validation methods
    - Add validate_file_exists() and validate_directory_exists() methods
    - Add handle_video_error() for graceful degradation
    - Add handle_model_error() for critical errors
    - Add safe_execute() wrapper for automatic error handling
    - Add success/error counting and get_summary() method
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 2. Create configuration files
  - [ ] 2.1 Create .env.example file with all environment variables
    - Add DATA_ROOT, MODEL_PATH, VIDEO_SOURCE, OUTPUT_DIR, DATASET_PATH, LOG_DIR
    - Add bilingual comments explaining each variable
    - Add example values for each variable
    - Document default values used when variables are not set
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 2.2 Create requirements.txt file with all dependencies
    - Add ultralytics, opencv-python, torch, torchvision
    - Add onnx, onnxruntime for ONNX support
    - Add python-dotenv for configuration management
    - Add fastapi, uvicorn, python-multipart for HTTP serving
    - Add numpy, Pillow, pyyaml utilities
    - Specify compatible version ranges for each dependency
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 2.3 Update data.yaml to use relative paths
    - Replace absolute paths with relative paths for train/val/test
    - Use ./dataset/uadetrac/ prefix for all paths
    - Keep class names and Roboflow metadata unchanged
    - _Requirements: 1.6_

- [ ] 3. Checkpoint - Verify infrastructure setup
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Refactor training scripts
  - [ ] 4.1 Refactor yolo.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before training
    - Replace hardcoded paths with config.data_root / "data.yaml"
    - Add comprehensive logging for all operations
    - Add error handling with descriptive messages
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.2, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4_

  - [ ] 4.2 Refactor vnyolo.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before training
    - Replace hardcoded paths with environment-based paths
    - Add comprehensive logging for all operations
    - Add error handling with descriptive messages
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.2, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4_

- [ ] 5. Refactor tracking and utility scripts
  - [ ] 5.1 Refactor tracking_yolo.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before processing
    - Replace hardcoded model path with config.model_path / "best.pt"
    - Support both single video file and directory of videos
    - Add graceful degradation for video processing errors
    - Add processing summary reporting (success/failed counts)
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.2, 4.4, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 9.1, 9.5_

  - [ ] 5.2 Refactor framecutter.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before processing
    - Replace hardcoded paths with config.video_source and config.output_dir
    - Add graceful degradation for video processing errors
    - Add processing summary reporting (success/failed counts)
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.3, 4.4, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 9.1, 9.2, 9.5_

  - [ ] 5.3 Refactor export_onnx.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before export
    - Replace hardcoded model path with config.model_path / "best.pt"
    - Add comprehensive logging for export operations
    - Add error handling for model loading and export
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.2, 4.5, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4_

  - [ ] 5.4 Refactor scriptprelabel.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before processing
    - Replace hardcoded paths with environment-based paths
    - Add comprehensive logging and error handling
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4_

  - [ ] 5.5 Refactor onnx_testing.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before testing
    - Replace hardcoded paths with environment-based paths
    - Add comprehensive logging and error handling
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4_

  - [ ] 5.6 Refactor benchmark_onnx.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before benchmarking
    - Replace hardcoded paths with environment-based paths
    - Add comprehensive logging and error handling
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4_

  - [ ] 5.7 Refactor chia.py to use infrastructure modules
    - Import ConfigManager, SystemLogger, ErrorHandler
    - Initialize configuration, logger, and error handler at startup
    - Validate configuration paths before processing
    - Replace hardcoded paths with environment-based paths
    - Add comprehensive logging and error handling
    - Add bilingual comments throughout the script
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.6, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4_

- [ ] 6. Checkpoint - Verify script refactoring
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Create Docker deployment components
  - [ ] 7.1 Create Dockerfile for containerized deployment
    - Use Python 3.11 slim base image
    - Install system dependencies for OpenCV (libgl1-mesa-glx, libglib2.0-0)
    - Create non-privileged user (appuser) for security
    - Copy and install Python dependencies from requirements.txt
    - Copy application source code
    - Create necessary directories (models, output, logs)
    - Set proper permissions for appuser
    - Expose port 8000 for HTTP serving
    - Set default environment variables
    - Define CMD to run serve_model.py
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 10.2_

  - [ ] 7.2 Create serve_model.py for HTTP model serving
    - Import FastAPI, YOLO, ConfigManager, SystemLogger, ErrorHandler
    - Initialize FastAPI application
    - Implement startup_event() to load model and validate paths
    - Implement /health endpoint for health checks
    - Implement /predict endpoint for image inference
    - Accept multipart/form-data image uploads
    - Return JSON response with predictions (class, class_name, confidence, bbox)
    - Add comprehensive error handling for invalid inputs
    - Add logging for all requests and errors
    - Exit with non-zero status if model loading fails
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 8. Create testing infrastructure
  - [ ]* 8.1 Write unit tests for config.py
    - Test loading valid .env files
    - Test default value fallback when environment variables are not set
    - Test invalid configuration detection
    - Test path validation for existing and missing paths
    - Test directory creation for output paths
    - Test permission checking for read/write operations
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.1, 5.2, 5.3, 5.4_

  - [ ]* 8.2 Write unit tests for logger.py
    - Test log file creation in specified directory
    - Test console output for log messages
    - Test timestamp formatting in log messages
    - Test INFO, WARNING, ERROR log levels
    - Test exception logging with stack traces
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 8.3 Write unit tests for error_handler.py
    - Test PathError, ModelError, VideoError exception raising
    - Test validate_file_exists() for existing and missing files
    - Test validate_directory_exists() with create=True and create=False
    - Test handle_video_error() graceful degradation
    - Test handle_model_error() critical error handling
    - Test safe_execute() wrapper for automatic error handling
    - Test get_summary() reporting
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 8.4 Write integration tests for Docker deployment
    - Test Dockerfile builds successfully without errors
    - Test container starts and loads model at startup
    - Test /health endpoint returns healthy status
    - Test /predict endpoint with valid image file
    - Test /predict endpoint with invalid image file
    - Test container exits with non-zero status if model is missing
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 8.5 Write property-based tests for configuration management
    - **Property 1: Environment Variable Retrieval with Fallback**
    - **Validates: Requirements 1.3, 1.4**
    - Test that for any environment variable, the system retrieves its value when defined or returns the default when not defined

  - [ ]* 8.6 Write property-based tests for path validation
    - **Property 4: Path Validation and Error Handling**
    - **Validates: Requirements 4.1, 4.2**
    - Test that for any file path, the system verifies existence and raises descriptive exceptions for missing paths

  - [ ]* 8.7 Write property-based tests for graceful degradation
    - **Property 6: Graceful Video Processing Degradation**
    - **Validates: Requirements 4.4, 9.1**
    - Test that for any batch of videos with some invalid files, the system logs errors and continues processing valid files

  - [ ]* 8.8 Write property-based tests for logging
    - **Property 13: Dual Logging Output**
    - **Validates: Requirements 6.2**
    - Test that for any log message, it appears in both console and log file

  - [ ]* 8.9 Write property-based tests for bilingual comments
    - **Property 17: Bilingual Comment Format**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    - Test that for any comment in the codebase, it follows the bilingual format (Vietnamese / English)

- [ ] 9. Create documentation
  - [ ] 9.1 Create README.md with setup and usage instructions
    - Add project overview and architecture description
    - Add installation instructions (pip install -r requirements.txt)
    - Add configuration instructions (.env file setup)
    - Add usage examples for training, tracking, and export scripts
    - Add Docker deployment instructions (build and run)
    - Add API usage examples (curl commands for /health and /predict)
    - Add troubleshooting section for common issues
    - Add bilingual documentation (Vietnamese and English)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 9.2 Create MIGRATION.md with migration guide
    - Add step-by-step migration instructions from old to new architecture
    - Document changes to existing scripts
    - Document new infrastructure modules
    - Document configuration file changes
    - Add testing instructions for verifying migration
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [ ] 10. Final checkpoint - Verify complete implementation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property-based tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The infrastructure modules (config.py, logger.py, error_handler.py) are foundational and must be completed first
- All existing scripts must be refactored to use the infrastructure modules
- Docker deployment enables production-ready model serving via HTTP API
- Bilingual comments support both Vietnamese and English-speaking developers

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3"] },
    { "id": 2, "tasks": ["4.1", "4.2"] },
    { "id": 3, "tasks": ["5.1", "5.2", "5.3", "5.4"] },
    { "id": 4, "tasks": ["5.5", "5.6", "5.7"] },
    { "id": 5, "tasks": ["7.1", "7.2"] },
    { "id": 6, "tasks": ["8.1", "8.2", "8.3", "8.4", "8.5", "8.6", "8.7", "8.8", "8.9"] },
    { "id": 7, "tasks": ["9.1", "9.2"] }
  ]
}
```
