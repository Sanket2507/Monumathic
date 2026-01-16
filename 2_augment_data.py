"""
2. AUGMENT DATA - Image Data Augmentation for Monument Dataset
Applies various transformations to increase dataset size and variety.
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

# Configuration
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
AUGMENTED_DIR = BASE_DIR / "dataset_augmented"
TARGET_IMAGES_PER_CLASS = 500  # Target images after augmentation


class ImageAugmentor:
    """Image augmentation class with various transformation methods."""
    
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.original = Image.open(image_path).convert("RGB")
        self.augmented_images = []
    
    def rotate(self, angles=[90, 180, 270]):
        """Rotate image by specified angles."""
        for angle in angles:
            rotated = self.original.rotate(angle, expand=True)
            self.augmented_images.append(("rotate", angle, rotated))
        return self
    
    def flip(self):
        """Horizontal and vertical flip."""
        h_flip = ImageOps.mirror(self.original)
        v_flip = ImageOps.flip(self.original)
        self.augmented_images.append(("hflip", 0, h_flip))
        self.augmented_images.append(("vflip", 0, v_flip))
        return self
    
    def brightness(self, factors=[0.7, 0.85, 1.15, 1.3]):
        """Adjust brightness."""
        enhancer = ImageEnhance.Brightness(self.original)
        for factor in factors:
            enhanced = enhancer.enhance(factor)
            self.augmented_images.append(("bright", int(factor*100), enhanced))
        return self
    
    def contrast(self, factors=[0.7, 0.85, 1.15, 1.3]):
        """Adjust contrast."""
        enhancer = ImageEnhance.Contrast(self.original)
        for factor in factors:
            enhanced = enhancer.enhance(factor)
            self.augmented_images.append(("contrast", int(factor*100), enhanced))
        return self
    
    def saturation(self, factors=[0.5, 0.75, 1.25, 1.5]):
        """Adjust color saturation."""
        enhancer = ImageEnhance.Color(self.original)
        for factor in factors:
            enhanced = enhancer.enhance(factor)
            self.augmented_images.append(("saturation", int(factor*100), enhanced))
        return self
    
    def sharpness(self, factors=[0.5, 1.5, 2.0]):
        """Adjust sharpness."""
        enhancer = ImageEnhance.Sharpness(self.original)
        for factor in factors:
            enhanced = enhancer.enhance(factor)
            self.augmented_images.append(("sharp", int(factor*100), enhanced))
        return self
    
    def blur(self, radii=[1, 2]):
        """Apply Gaussian blur."""
        for radius in radii:
            blurred = self.original.filter(ImageFilter.GaussianBlur(radius))
            self.augmented_images.append(("blur", radius, blurred))
        return self
    
    def crop_center(self, crop_percentages=[0.85, 0.9]):
        """Center crop the image."""
        w, h = self.original.size
        for pct in crop_percentages:
            new_w, new_h = int(w * pct), int(h * pct)
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            cropped = self.original.crop((left, top, left + new_w, top + new_h))
            self.augmented_images.append(("crop", int(pct*100), cropped))
        return self
    
    def random_crop(self, num_crops=3, crop_pct=0.8):
        """Random crops from different positions."""
        w, h = self.original.size
        crop_w, crop_h = int(w * crop_pct), int(h * crop_pct)
        
        for i in range(num_crops):
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            cropped = self.original.crop((left, top, left + crop_w, top + crop_h))
            self.augmented_images.append(("rcrop", i, cropped))
        return self
    
    def perspective_transform(self):
        """Apply slight perspective distortion."""
        w, h = self.original.size
        
        # Define transformation coefficients for slight skew
        coefficients = [
            (1.05, 0.1, -w*0.05, 0, 1, 0, 0, 0),  # Slight right skew
            (1.05, -0.1, 0, 0, 1, 0, 0, 0),        # Slight left skew
        ]
        
        for i, coeff in enumerate(coefficients):
            try:
                transformed = self.original.transform(
                    (w, h), Image.AFFINE, coeff[:6], Image.BICUBIC
                )
                self.augmented_images.append(("persp", i, transformed))
            except Exception:
                pass
        return self
    
    def add_noise(self, noise_levels=[10, 20]):
        """Add random noise to image."""
        img_array = np.array(self.original)
        
        for level in noise_levels:
            noise = np.random.normal(0, level, img_array.shape).astype(np.int16)
            noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            noisy_img = Image.fromarray(noisy)
            self.augmented_images.append(("noise", level, noisy_img))
        return self
    
    def grayscale(self):
        """Convert to grayscale (3-channel)."""
        gray = ImageOps.grayscale(self.original)
        gray_rgb = Image.merge("RGB", (gray, gray, gray))
        self.augmented_images.append(("gray", 0, gray_rgb))
        return self
    
    def sepia(self):
        """Apply sepia tone filter."""
        img_array = np.array(self.original, dtype=np.float32)
        
        # Sepia matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        sepia_img = img_array @ sepia_matrix.T
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        self.augmented_images.append(("sepia", 0, Image.fromarray(sepia_img)))
        return self
    
    def apply_all_augmentations(self):
        """Apply all augmentation techniques."""
        self.rotate(angles=[90, 180, 270])
        self.flip()
        self.brightness(factors=[0.7, 1.3])
        self.contrast(factors=[0.7, 1.3])
        self.saturation(factors=[0.6, 1.4])
        self.sharpness(factors=[0.5, 1.5])
        self.blur(radii=[1])
        self.crop_center(crop_percentages=[0.85])
        self.random_crop(num_crops=2, crop_pct=0.8)
        self.add_noise(noise_levels=[15])
        self.grayscale()
        self.sepia()
        return self
    
    def apply_basic_augmentations(self):
        """Apply basic augmentation (faster, fewer variations)."""
        self.rotate(angles=[90, 180, 270])
        self.flip()
        self.brightness(factors=[0.8, 1.2])
        self.contrast(factors=[0.8, 1.2])
        return self
    
    def save_augmented(self, output_dir, base_name=None):
        """Save all augmented images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if base_name is None:
            base_name = self.image_path.stem
        
        saved_count = 0
        
        # Save original
        original_path = output_dir / f"{base_name}_original.jpg"
        self.original.save(original_path, "JPEG", quality=90)
        saved_count += 1
        
        # Save augmented versions
        for aug_type, param, img in self.augmented_images:
            filename = f"{base_name}_{aug_type}_{param}.jpg"
            filepath = output_dir / filename
            img.save(filepath, "JPEG", quality=85)
            saved_count += 1
        
        return saved_count


def augment_single_image(args):
    """Augment a single image (for parallel processing)."""
    image_path, output_dir, mode = args
    try:
        augmentor = ImageAugmentor(image_path)
        
        if mode == "full":
            augmentor.apply_all_augmentations()
        else:
            augmentor.apply_basic_augmentations()
        
        count = augmentor.save_augmented(output_dir)
        return image_path.name, count
    except Exception as e:
        return image_path.name, f"ERROR: {e}"


def augment_monument_class(monument_folder, mode="basic"):
    """Augment all images for a single monument class."""
    input_dir = DATASET_DIR / monument_folder
    output_dir = AUGMENTED_DIR / monument_folder
    
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"⚠️ No images found in: {input_dir}")
        return 0
    
    print(f"\n🏛️ Augmenting: {monument_folder}")
    print(f"   Source images: {len(image_files)}")
    
    total_augmented = 0
    
    # Process images in parallel
    args_list = [(img, output_dir, mode) for img in image_files]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(augment_single_image, args_list))
    
    for filename, count in results:
        if isinstance(count, int):
            total_augmented += count
        else:
            print(f"   ❌ {filename}: {count}")
    
    print(f"   ✅ Generated {total_augmented} augmented images")
    
    return total_augmented


def augment_all_monuments(mode="basic"):
    """Augment images for all monument classes."""
    print("=" * 60)
    print("🖼️ MONUMENT DATA AUGMENTATION")
    print("=" * 60)
    
    if not DATASET_DIR.exists():
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        print("   Run 1_fetch_images.py first to collect images.")
        return
    
    # Create augmented dataset directory
    AUGMENTED_DIR.mkdir(exist_ok=True)
    
    # Get all monument folders
    monument_folders = [f.name for f in DATASET_DIR.iterdir() if f.is_dir()]
    
    if not monument_folders:
        print("❌ No monument folders found in dataset.")
        return
    
    print(f"\n📁 Found {len(monument_folders)} monument classes")
    print(f"🔄 Augmentation mode: {mode}")
    
    results = {}
    total_images = 0
    
    for folder in monument_folders:
        count = augment_monument_class(folder, mode)
        results[folder] = count
        total_images += count
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 AUGMENTATION SUMMARY")
    print("=" * 60)
    
    for monument, count in sorted(results.items()):
        status = "✅" if count >= 100 else "⚠️" if count >= 50 else "❌"
        print(f"{status} {monument}: {count} images")
    
    print(f"\n📦 Total augmented images: {total_images}")
    print(f"📁 Output location: {AUGMENTED_DIR}")
    
    # Save metadata
    metadata = {
        "monuments": results,
        "total_images": total_images,
        "augmented_path": str(AUGMENTED_DIR),
        "mode": mode
    }
    
    with open(BASE_DIR / "augmentation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def verify_augmented_dataset():
    """Verify the augmented dataset statistics."""
    print("\n📊 AUGMENTED DATASET VERIFICATION")
    print("=" * 60)
    
    if not AUGMENTED_DIR.exists():
        print("❌ Augmented dataset not found. Run augmentation first.")
        return
    
    total = 0
    classes = []
    
    for folder in sorted(AUGMENTED_DIR.iterdir()):
        if folder.is_dir():
            count = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))
            classes.append((folder.name, count))
            total += count
    
    for name, count in classes:
        status = "✅" if count >= TARGET_IMAGES_PER_CLASS else "⚠️"
        print(f"{status} {name}: {count} images")
    
    print(f"\n📦 Total: {total} images across {len(classes)} classes")
    print(f"🎯 Target per class: {TARGET_IMAGES_PER_CLASS}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monument Image Augmentation")
    parser.add_argument("--mode", choices=["basic", "full"], default="basic",
                        help="Augmentation mode: basic (faster) or full (more variations)")
    parser.add_argument("--monument", type=str, help="Augment specific monument folder")
    parser.add_argument("--verify", action="store_true", help="Verify augmented dataset")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_augmented_dataset()
    elif args.monument:
        augment_monument_class(args.monument, args.mode)
    else:
        augment_all_monuments(args.mode)
