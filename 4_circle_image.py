"""
4. CIRCLE IMAGE - Monument Detection Visualization
Draws circles, bounding boxes, and annotations around detected monuments.
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

import torch
from torchvision import transforms, models
import torch.nn as nn

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "circled_output"
IMG_SIZE = 128

# Colors for visualization (RGB)
COLORS = {
    "primary": (0, 200, 83),      # Green
    "secondary": (255, 87, 34),    # Orange
    "accent": (33, 150, 243),      # Blue
    "highlight": (255, 235, 59),   # Yellow
    "error": (244, 67, 54),        # Red
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MonumentVisualizer:
    """Class to visualize monument detection results."""
    
    def __init__(self, model_path=None, class_names_path=None):
        self.model = None
        self.class_names = []
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load model if paths provided
        if model_path and class_names_path:
            self.load_model(model_path, class_names_path)
    
    def load_model(self, model_path, class_names_path):
        """Load trained model and class names."""
        model_path = Path(model_path)
        class_names_path = Path(class_names_path)
        
        # Load class names
        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)
        
        num_classes = len(self.class_names)
        
        # Create and load model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Try loading with different fc configurations
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except RuntimeError:
            # Fallback: simple fc layer
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        self.model.to(DEVICE)
        self.model.eval()
        
        print(f"✅ Model loaded: {model_path}")
        print(f"   Classes: {num_classes}")
    
    def predict(self, image):
        """Predict monument from image."""
        if self.model is None:
            return "Unknown", 0.0, []
        
        # Preprocess
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        img_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(self.class_names)))
        top5 = [
            (self.class_names[idx], prob.item())
            for idx, prob in zip(top5_indices[0], top5_probs[0])
        ]
        
        return predicted_class, confidence_score, top5
    
    def draw_circle_highlight(self, image, center=None, radius=None, color=COLORS["primary"], 
                             width=5, glow=True):
        """Draw a circle highlight on the image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGBA")
        else:
            image = image.convert("RGBA")
        
        draw = ImageDraw.Draw(image)
        
        # Default center and radius
        w, h = image.size
        if center is None:
            center = (w // 2, h // 2)
        if radius is None:
            radius = min(w, h) // 3
        
        # Draw glow effect
        if glow:
            for i in range(10, 0, -2):
                glow_color = (*color, int(255 * (i / 10) * 0.3))
                draw.ellipse(
                    [center[0] - radius - i, center[1] - radius - i,
                     center[0] + radius + i, center[1] + radius + i],
                    outline=glow_color, width=width + i
                )
        
        # Draw main circle
        draw.ellipse(
            [center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius],
            outline=(*color, 255), width=width
        )
        
        return image.convert("RGB")
    
    def draw_bounding_box(self, image, bbox=None, color=COLORS["primary"], 
                          width=4, label=None, confidence=None):
        """Draw a bounding box with optional label."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.copy()
        
        draw = ImageDraw.Draw(image)
        w, h = image.size
        
        # Default bounding box (80% of image)
        if bbox is None:
            margin = int(min(w, h) * 0.1)
            bbox = [margin, margin, w - margin, h - margin]
        
        # Draw box
        draw.rectangle(bbox, outline=color, width=width)
        
        # Draw corner accents
        corner_length = 20
        corners = [
            [(bbox[0], bbox[1]), (bbox[0] + corner_length, bbox[1]), (bbox[0], bbox[1] + corner_length)],  # Top-left
            [(bbox[2] - corner_length, bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[1] + corner_length)],  # Top-right
            [(bbox[0], bbox[3] - corner_length), (bbox[0], bbox[3]), (bbox[0] + corner_length, bbox[3])],  # Bottom-left
            [(bbox[2], bbox[3] - corner_length), (bbox[2], bbox[3]), (bbox[2] - corner_length, bbox[3])]   # Bottom-right
        ]
        
        for corner in corners:
            draw.line([corner[0], corner[1]], fill=color, width=width + 2)
            draw.line([corner[0], corner[2]], fill=color, width=width + 2)
        
        # Draw label
        if label:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            label_text = label
            if confidence is not None:
                label_text = f"{label} ({confidence*100:.1f}%)"
            
            # Label background
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            label_x = bbox[0]
            label_y = bbox[1] - text_h - 10
            if label_y < 0:
                label_y = bbox[3] + 5
            
            # Background rectangle
            draw.rectangle(
                [label_x - 5, label_y - 5, label_x + text_w + 10, label_y + text_h + 5],
                fill=color
            )
            
            # Text
            draw.text((label_x, label_y), label_text, fill=(255, 255, 255), font=font)
        
        return image
    
    def draw_confidence_bar(self, image, confidence, position="bottom", color=COLORS["primary"]):
        """Draw a confidence bar on the image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.copy()
        
        draw = ImageDraw.Draw(image)
        w, h = image.size
        
        bar_height = 20
        bar_margin = 20
        
        if position == "bottom":
            bar_y = h - bar_height - bar_margin
        else:
            bar_y = bar_margin
        
        # Background bar
        draw.rectangle(
            [bar_margin, bar_y, w - bar_margin, bar_y + bar_height],
            fill=(50, 50, 50)
        )
        
        # Confidence bar
        bar_width = int((w - 2 * bar_margin) * confidence)
        draw.rectangle(
            [bar_margin, bar_y, bar_margin + bar_width, bar_y + bar_height],
            fill=color
        )
        
        # Percentage text
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        text = f"{confidence*100:.1f}%"
        draw.text((w - bar_margin - 50, bar_y + 2), text, fill=(255, 255, 255), font=font)
        
        return image
    
    def draw_top5_predictions(self, image, top5, position="right"):
        """Draw top 5 predictions sidebar."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.copy()
        
        w, h = image.size
        
        # Create new image with sidebar
        sidebar_width = 250
        new_image = Image.new("RGB", (w + sidebar_width, h), (30, 30, 30))
        new_image.paste(image, (0, 0))
        
        draw = ImageDraw.Draw(new_image)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 18)
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            font = ImageFont.load_default()
        
        # Title
        draw.text((w + 10, 20), "Top Predictions:", fill=(255, 255, 255), font=title_font)
        
        # Predictions
        y_pos = 60
        colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], 
                       COLORS["highlight"], (150, 150, 150)]
        
        for i, (name, prob) in enumerate(top5):
            color = colors_list[i] if i < len(colors_list) else (150, 150, 150)
            
            # Bar
            bar_width = int(200 * prob)
            draw.rectangle([w + 10, y_pos, w + 10 + bar_width, y_pos + 20], fill=color)
            
            # Text
            display_name = name.replace("_", " ")[:20]
            draw.text((w + 15, y_pos + 2), f"{i+1}. {display_name}", fill=(255, 255, 255), font=font)
            draw.text((w + 180, y_pos + 2), f"{prob*100:.1f}%", fill=(255, 255, 255), font=font)
            
            y_pos += 35
        
        return new_image
    
    def visualize_detection(self, image_path, output_path=None, style="full"):
        """
        Full visualization pipeline for monument detection.
        
        Styles:
        - "full": Circle + box + confidence bar + top5 sidebar
        - "simple": Just bounding box with label
        - "circle": Circle highlight only
        - "minimal": Box only, no labels
        """
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        # Get prediction
        monument, confidence, top5 = self.predict(image)
        
        print(f"🔍 Detected: {monument} ({confidence*100:.1f}%)")
        
        # Choose color based on confidence
        if confidence > 0.8:
            color = COLORS["primary"]  # Green
        elif confidence > 0.5:
            color = COLORS["highlight"]  # Yellow
        else:
            color = COLORS["error"]  # Red
        
        if style == "full":
            # Draw bounding box
            result = self.draw_bounding_box(image, color=color, label=monument, confidence=confidence)
            # Add confidence bar
            result = self.draw_confidence_bar(result, confidence, color=color)
            # Add top5 sidebar
            result = self.draw_top5_predictions(result, top5)
            
        elif style == "simple":
            result = self.draw_bounding_box(image, color=color, label=monument, confidence=confidence)
            
        elif style == "circle":
            result = self.draw_circle_highlight(image, color=color)
            
        elif style == "minimal":
            result = self.draw_bounding_box(image, color=color)
            
        else:
            result = image
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path, quality=95)
            print(f"💾 Saved: {output_path}")
        
        return result, monument, confidence, top5


def visualize_single_image(image_path, model_path=None, class_names_path=None, style="full"):
    """Visualize a single image."""
    
    # Use default paths if not provided
    if model_path is None:
        model_path = MODEL_DIR / "fast_monument_cnn.pth"
    if class_names_path is None:
        class_names_path = MODEL_DIR / "class_names.json"
    
    # Check files exist
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run 3_train_model.py first to train the model.")
        return None
    
    if not Path(class_names_path).exists():
        print(f"❌ Class names not found: {class_names_path}")
        return None
    
    # Create visualizer
    visualizer = MonumentVisualizer(model_path, class_names_path)
    
    # Generate output path
    image_path = Path(image_path)
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_detected{image_path.suffix}"
    
    # Visualize
    result, monument, confidence, top5 = visualizer.visualize_detection(
        image_path, output_path, style=style
    )
    
    return result


def visualize_batch(image_folder, model_path=None, class_names_path=None, style="simple"):
    """Visualize all images in a folder."""
    
    image_folder = Path(image_folder)
    if not image_folder.exists():
        print(f"❌ Folder not found: {image_folder}")
        return
    
    # Use default paths if not provided
    if model_path is None:
        model_path = MODEL_DIR / "fast_monument_cnn.pth"
    if class_names_path is None:
        class_names_path = MODEL_DIR / "class_names.json"
    
    # Create visualizer
    visualizer = MonumentVisualizer(model_path, class_names_path)
    
    # Get all images
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No images found in: {image_folder}")
        return
    
    print(f"📁 Processing {len(image_files)} images from {image_folder}")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    batch_output = OUTPUT_DIR / image_folder.name
    batch_output.mkdir(exist_ok=True)
    
    results = []
    
    for img_path in image_files:
        output_path = batch_output / f"{img_path.stem}_detected{img_path.suffix}"
        
        try:
            result, monument, confidence, top5 = visualizer.visualize_detection(
                img_path, output_path, style=style
            )
            results.append({
                "image": img_path.name,
                "monument": monument,
                "confidence": confidence,
                "top5": top5
            })
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
            continue
    
    # Save results summary
    summary_path = batch_output / "detection_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Processed {len(results)} images")
    print(f"📁 Output: {batch_output}")
    print(f"📋 Summary: {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monument Detection Visualization")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--classes", type=str, help="Path to class names JSON")
    parser.add_argument("--style", choices=["full", "simple", "circle", "minimal"],
                        default="full", help="Visualization style")
    
    args = parser.parse_args()
    
    if args.image:
        visualize_single_image(args.image, args.model, args.classes, args.style)
    elif args.folder:
        visualize_batch(args.folder, args.model, args.classes, args.style)
    else:
        print("Usage:")
        print("  python 4_circle_image.py --image path/to/image.jpg")
        print("  python 4_circle_image.py --folder path/to/images/")
        print("\nOptions:")
        print("  --style full|simple|circle|minimal")
