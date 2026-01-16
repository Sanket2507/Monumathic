"""
Monumathic - Monument Detection Module
======================================

A complete pipeline for monument image classification.

Scripts:
    1_fetch_images.py   - Download monument images from APIs
    2_augment_data.py   - Augment images for better training
    3_train_model.py    - Train ResNet-based classifier
    4_circle_image.py   - Visualize detections with circles/boxes
    5_predict.py        - Production inference

Quick Start:
    # Step 1: Fetch images (requires API keys)
    python 1_fetch_images.py --create-dirs
    
    # Step 2: Augment dataset
    python 2_augment_data.py --mode basic
    
    # Step 3: Train model
    python 3_train_model.py --epochs 50
    
    # Step 4: Visualize detection
    python 4_circle_image.py --image test.jpg --style full
    
    # Step 5: Predict
    python 5_predict.py --image test.jpg
    python 5_predict.py --interactive

For API integration, use:
    from Monumathic.predict import MonumentPredictor, predict_monument
    
    result = predict_monument(image_bytes)
    print(result['monument'], result['confidence'])
"""

__version__ = "1.0.0"
__author__ = "Trend Tripper Team"

# Expose main classes for easy import
# Note: Module names starting with numbers require importlib
try:
    import importlib
    _predict_module = importlib.import_module(".5_predict", package="Monumathic")
    MonumentPredictor = _predict_module.MonumentPredictor
    predict_monument = _predict_module.predict_monument
    get_predictor = _predict_module.get_predictor
except (ImportError, ModuleNotFoundError):
    # Module not available (likely missing dependencies)
    pass
