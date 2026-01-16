"""
5. PREDICT - Monument Prediction & Inference
Production-ready inference for monument image classification.
"""

import os
import io
import json
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR.parent / "data"  # Parent backend/data folder
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PredictionResult:
    """Data class for prediction results."""
    monument_name: str
    confidence: float
    top_predictions: List[Tuple[str, float]]
    description: Optional[str] = None
    history: Optional[List[dict]] = None
    error: Optional[str] = None


class MonumentPredictor:
    """
    Monument image prediction class.
    Production-ready inference for monument classification.
    """
    
    def __init__(self, model_path: str = None, class_names_path: str = None):
        self.model = None
        self.class_names = []
        self.monument_info = {}
        self.monument_history = {}
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model if paths provided
        if model_path and class_names_path:
            self.load_model(model_path, class_names_path)
        else:
            self._auto_load_model()
        
        # Load monument information
        self._load_monument_data()
    
    def _auto_load_model(self):
        """Automatically load model from default locations."""
        # Try different model paths
        possible_paths = [
            MODEL_DIR / "fast_monument_cnn.pth",
            BASE_DIR.parent / "model" / "fast_monument_cnn.pth",
            BASE_DIR / "fast_monument_cnn.pth",
        ]
        
        class_name_paths = [
            MODEL_DIR / "class_names.json",
            BASE_DIR / "class_names.json",
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        class_names_path = None
        for path in class_name_paths:
            if path.exists():
                class_names_path = path
                break
        
        if model_path and class_names_path:
            self.load_model(model_path, class_names_path)
        else:
            print(f"⚠️ Model not found. Looked in: {possible_paths}")
            print("   Run 3_train_model.py first to train the model.")
    
    def load_model(self, model_path: Union[str, Path], class_names_path: Union[str, Path]):
        """Load trained model and class names."""
        model_path = Path(model_path)
        class_names_path = Path(class_names_path)
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        if not class_names_path.exists():
            print(f"❌ Class names file not found: {class_names_path}")
            return False
        
        # Load class names
        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)
        
        num_classes = len(self.class_names)
        
        # Create model architecture
        self.model = models.resnet18(pretrained=False)
        
        # Try different FC layer configurations
        fc_configs = [
            # Full custom FC (from training)
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            ),
            # Simple FC
            nn.Linear(self.model.fc.in_features, num_classes),
        ]
        
        loaded = False
        for fc_config in fc_configs:
            try:
                self.model.fc = fc_config
                state_dict = torch.load(model_path, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                loaded = True
                break
            except RuntimeError:
                continue
        
        if not loaded:
            print(f"❌ Failed to load model weights from {model_path}")
            return False
        
        self.model.to(DEVICE)
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Path: {model_path}")
        print(f"   Classes: {num_classes}")
        print(f"   Device: {DEVICE}")
        
        return True
    
    def _load_monument_data(self):
        """Load monument information and history."""
        # Monument info
        info_paths = [
            DATA_DIR / "monument_info.json",
            BASE_DIR.parent / "data" / "monument_info.json",
        ]
        
        for path in info_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.monument_info = json.load(f)
                print(f"✅ Loaded monument info: {len(self.monument_info)} entries")
                break
        
        # Monument history
        history_paths = [
            DATA_DIR / "monument_history.json",
            BASE_DIR.parent / "data" / "monument_history.json",
        ]
        
        for path in history_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.monument_history = json.load(f)
                print(f"✅ Loaded monument history: {len(self.monument_history)} entries")
                break
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, bytes]) -> torch.Tensor:
        """Preprocess image for inference."""
        
        # Handle different input types
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(DEVICE)
    
    def predict(self, image: Union[str, Path, Image.Image, bytes], 
                top_k: int = 5) -> PredictionResult:
        """
        Predict monument from image.
        
        Args:
            image: Image path, PIL Image, or bytes
            top_k: Number of top predictions to return
        
        Returns:
            PredictionResult with monument name, confidence, and additional info
        """
        
        if self.model is None:
            return PredictionResult(
                monument_name="Unknown",
                confidence=0.0,
                top_predictions=[],
                error="Model not loaded"
            )
        
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
                
                # Best prediction
                best_idx = top_indices[0][0].item()
                best_prob = top_probs[0][0].item()
                monument_name = self.class_names[best_idx]
                
                # Top-k list
                top_predictions = [
                    (self.class_names[idx.item()], prob.item())
                    for idx, prob in zip(top_indices[0], top_probs[0])
                ]
            
            # Get monument info
            # Try different name formats
            description = None
            history = None
            
            name_variants = [
                monument_name,
                monument_name.replace("_", " "),
                monument_name.split("_")[0],
            ]
            
            for name in name_variants:
                if name in self.monument_info:
                    description = self.monument_info[name]
                    break
            
            for name in name_variants:
                if name in self.monument_history:
                    history = self.monument_history[name]
                    break
            
            return PredictionResult(
                monument_name=monument_name.replace("_", " "),
                confidence=best_prob,
                top_predictions=top_predictions,
                description=description,
                history=history
            )
            
        except Exception as e:
            return PredictionResult(
                monument_name="Unknown",
                confidence=0.0,
                top_predictions=[],
                error=str(e)
            )
    
    def predict_from_url(self, url: str, top_k: int = 5) -> PredictionResult:
        """Predict monument from image URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return self.predict(response.content, top_k)
        except Exception as e:
            return PredictionResult(
                monument_name="Unknown",
                confidence=0.0,
                top_predictions=[],
                error=f"Failed to download image: {e}"
            )
    
    def predict_batch(self, images: List[Union[str, Path, Image.Image]], 
                      top_k: int = 5) -> List[PredictionResult]:
        """Predict monuments from multiple images."""
        return [self.predict(img, top_k) for img in images]
    
    def get_monument_info(self, monument_name: str) -> dict:
        """Get detailed information about a monument."""
        # Normalize name
        name = monument_name.replace("_", " ")
        
        description = self.monument_info.get(name, "No information available.")
        history = self.monument_history.get(name, [])
        
        return {
            "name": name,
            "description": description,
            "history": history
        }


# Global predictor instance for API usage
_predictor = None


def get_predictor() -> MonumentPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MonumentPredictor()
    return _predictor


def predict_monument(image: Union[str, Path, Image.Image, bytes]) -> dict:
    """
    Simple prediction function for API integration.
    
    Args:
        image: Image path, PIL Image, or bytes
    
    Returns:
        Dictionary with prediction results
    """
    predictor = get_predictor()
    result = predictor.predict(image)
    
    return {
        "monument": result.monument_name,
        "confidence": result.confidence,
        "top_predictions": [
            {"name": name.replace("_", " "), "confidence": conf}
            for name, conf in result.top_predictions
        ],
        "description": result.description,
        "history": result.history,
        "error": result.error
    }


def predict_from_file(image_path: str) -> None:
    """Command-line prediction from file."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n🖼️ Predicting: {image_path}")
    print("=" * 50)
    
    predictor = get_predictor()
    result = predictor.predict(image_path)
    
    if result.error:
        print(f"❌ Error: {result.error}")
        return
    
    print(f"\n🏛️ Monument: {result.monument_name}")
    print(f"📊 Confidence: {result.confidence * 100:.1f}%")
    
    print(f"\n📋 Top Predictions:")
    for i, (name, conf) in enumerate(result.top_predictions, 1):
        bar = "█" * int(conf * 20)
        print(f"   {i}. {name.replace('_', ' ')}: {conf*100:.1f}% {bar}")
    
    if result.description:
        print(f"\n📖 Description:")
        print(f"   {result.description[:200]}..." if len(result.description) > 200 else f"   {result.description}")
    
    if result.history:
        print(f"\n📅 Historical Timeline: {len(result.history)} events")


def interactive_mode():
    """Interactive prediction mode."""
    print("\n" + "=" * 50)
    print("🏛️ MONUMENT PREDICTOR - Interactive Mode")
    print("=" * 50)
    print("Enter image path or URL to predict. Type 'quit' to exit.\n")
    
    predictor = get_predictor()
    
    while True:
        try:
            user_input = input("📷 Image path/URL: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Check if URL or file path
            if user_input.startswith(('http://', 'https://')):
                result = predictor.predict_from_url(user_input)
            else:
                if not Path(user_input).exists():
                    print(f"❌ File not found: {user_input}")
                    continue
                result = predictor.predict(user_input)
            
            if result.error:
                print(f"❌ Error: {result.error}")
                continue
            
            print(f"\n✅ Detected: {result.monument_name} ({result.confidence*100:.1f}%)")
            print("   Top 3:", ", ".join([f"{n.replace('_',' ')}({c*100:.0f}%)" 
                                          for n, c in result.top_predictions[:3]]))
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def benchmark_model(test_folder: str):
    """Benchmark model on a test folder."""
    test_folder = Path(test_folder)
    
    if not test_folder.exists():
        print(f"❌ Folder not found: {test_folder}")
        return
    
    print(f"\n🔬 Benchmarking on: {test_folder}")
    print("=" * 50)
    
    predictor = get_predictor()
    
    # Get all images
    image_files = list(test_folder.glob("**/*.jpg")) + \
                  list(test_folder.glob("**/*.png")) + \
                  list(test_folder.glob("**/*.jpeg"))
    
    if not image_files:
        print("❌ No images found")
        return
    
    print(f"📁 Found {len(image_files)} images\n")
    
    results = []
    correct = 0
    total = 0
    
    import time
    start_time = time.time()
    
    for img_path in image_files:
        # Get ground truth from folder name
        ground_truth = img_path.parent.name.replace("_", " ").lower()
        
        result = predictor.predict(img_path)
        predicted = result.monument_name.lower()
        
        is_correct = ground_truth in predicted or predicted in ground_truth
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "image": img_path.name,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "confidence": result.confidence,
            "correct": is_correct
        })
    
    elapsed = time.time() - start_time
    
    print(f"\n📊 BENCHMARK RESULTS")
    print(f"   Total images: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {correct/total*100:.1f}%")
    print(f"   Time: {elapsed:.2f}s ({elapsed/total*1000:.1f}ms per image)")
    
    # Save results
    results_path = test_folder / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "accuracy": correct/total,
            "total": total,
            "correct": correct,
            "time_seconds": elapsed,
            "results": results
        }, f, indent=2)
    print(f"   Results saved: {results_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monument Prediction")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--url", type=str, help="URL of image")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--benchmark", type=str, help="Benchmark on test folder")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--classes", type=str, help="Path to class names JSON")
    
    args = parser.parse_args()
    
    # Custom model paths
    if args.model and args.classes:
        _predictor = MonumentPredictor(args.model, args.classes)
    
    if args.image:
        predict_from_file(args.image)
    elif args.url:
        predictor = get_predictor()
        result = predictor.predict_from_url(args.url)
        print(f"\n🏛️ Monument: {result.monument_name}")
        print(f"📊 Confidence: {result.confidence * 100:.1f}%")
    elif args.benchmark:
        benchmark_model(args.benchmark)
    elif args.interactive:
        interactive_mode()
    else:
        # Default: show usage
        print("Monument Prediction Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  python 5_predict.py --image path/to/image.jpg")
        print("  python 5_predict.py --url https://example.com/image.jpg")
        print("  python 5_predict.py --interactive")
        print("  python 5_predict.py --benchmark path/to/test/folder")
        print("\nOptions:")
        print("  --model PATH    Custom model file")
        print("  --classes PATH  Custom class names JSON")
