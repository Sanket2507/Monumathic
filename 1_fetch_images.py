"""
1. FETCH IMAGES - Monument Image Downloader
Downloads monument images from various sources for training dataset.
"""

import os
import requests
import time
import hashlib
from pathlib import Path
from urllib.parse import quote_plus
import json

# Configuration
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_PER_MONUMENT = 100  # Target images per monument

# List of Indian monuments to fetch
MONUMENTS = [
    "Taj Mahal",
    "Red Fort Delhi",
    "Qutub Minar",
    "India Gate",
    "Gateway of India Mumbai",
    "Hawa Mahal Jaipur",
    "Mysore Palace",
    "Charminar Hyderabad",
    "Victoria Memorial Kolkata",
    "Golden Temple Amritsar",
    "Lotus Temple Delhi",
    "Meenakshi Temple Madurai",
    "Ajanta Caves",
    "Ellora Caves",
    "Konark Sun Temple",
    "Khajuraho Temples",
    "Hampi Ruins",
    "Fatehpur Sikri",
    "Sanchi Stupa",
    "Brihadeeswarar Temple",
    "Jantar Mantar Jaipur",
    "Humayun Tomb Delhi",
    "Amer Fort Jaipur",
    "Jaisalmer Fort",
    "Mehrangarh Fort Jodhpur",
    "Chittorgarh Fort",
    "Gwalior Fort",
    "Agra Fort",
    "Golconda Fort Hyderabad",
    "Udaipur City Palace",
]


def create_directories():
    """Create dataset directories for each monument."""
    DATASET_DIR.mkdir(exist_ok=True)
    
    for monument in MONUMENTS:
        # Create safe folder name
        folder_name = monument.replace(" ", "_").replace("/", "_")
        monument_dir = DATASET_DIR / folder_name
        monument_dir.mkdir(exist_ok=True)
        print(f"✅ Created directory: {monument_dir}")
    
    return True


def get_image_hash(image_data):
    """Generate hash for image to avoid duplicates."""
    return hashlib.md5(image_data).hexdigest()


def download_from_unsplash(monument_name, save_dir, count=30):
    """
    Download images from Unsplash (requires API key).
    Get your API key from: https://unsplash.com/developers
    """
    API_KEY = os.environ.get("UNSPLASH_API_KEY", "")
    if not API_KEY:
        print(f"⚠️ UNSPLASH_API_KEY not set. Skipping Unsplash downloads.")
        return 0
    
    downloaded = 0
    existing_hashes = set()
    
    # Get existing image hashes
    for img_file in save_dir.glob("*.jpg"):
        with open(img_file, "rb") as f:
            existing_hashes.add(get_image_hash(f.read()))
    
    try:
        url = f"https://api.unsplash.com/search/photos"
        params = {
            "query": monument_name,
            "per_page": min(count, 30),
            "orientation": "squarish"
        }
        headers = {"Authorization": f"Client-ID {API_KEY}"}
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"❌ Unsplash API error: {response.status_code}")
            return 0
        
        data = response.json()
        
        for i, photo in enumerate(data.get("results", [])):
            try:
                img_url = photo["urls"]["regular"]
                img_response = requests.get(img_url, timeout=30)
                
                if img_response.status_code == 200:
                    img_data = img_response.content
                    img_hash = get_image_hash(img_data)
                    
                    if img_hash not in existing_hashes:
                        filename = save_dir / f"unsplash_{i}_{img_hash[:8]}.jpg"
                        with open(filename, "wb") as f:
                            f.write(img_data)
                        existing_hashes.add(img_hash)
                        downloaded += 1
                        print(f"  📥 Downloaded: {filename.name}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error downloading image: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Unsplash fetch error: {e}")
    
    return downloaded


def download_from_pixabay(monument_name, save_dir, count=30):
    """
    Download images from Pixabay (requires API key).
    Get your API key from: https://pixabay.com/api/docs/
    """
    API_KEY = os.environ.get("PIXABAY_API_KEY", "")
    if not API_KEY:
        print(f"⚠️ PIXABAY_API_KEY not set. Skipping Pixabay downloads.")
        return 0
    
    downloaded = 0
    existing_hashes = set()
    
    # Get existing image hashes
    for img_file in save_dir.glob("*.jpg"):
        with open(img_file, "rb") as f:
            existing_hashes.add(get_image_hash(f.read()))
    
    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": API_KEY,
            "q": monument_name,
            "image_type": "photo",
            "per_page": min(count, 200),
            "safesearch": "true"
        }
        
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"❌ Pixabay API error: {response.status_code}")
            return 0
        
        data = response.json()
        
        for i, photo in enumerate(data.get("hits", [])):
            try:
                img_url = photo["webformatURL"]
                img_response = requests.get(img_url, timeout=30)
                
                if img_response.status_code == 200:
                    img_data = img_response.content
                    img_hash = get_image_hash(img_data)
                    
                    if img_hash not in existing_hashes:
                        filename = save_dir / f"pixabay_{i}_{img_hash[:8]}.jpg"
                        with open(filename, "wb") as f:
                            f.write(img_data)
                        existing_hashes.add(img_hash)
                        downloaded += 1
                        print(f"  📥 Downloaded: {filename.name}")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error downloading image: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Pixabay fetch error: {e}")
    
    return downloaded


def download_from_pexels(monument_name, save_dir, count=30):
    """
    Download images from Pexels (requires API key).
    Get your API key from: https://www.pexels.com/api/
    """
    API_KEY = os.environ.get("PEXELS_API_KEY", "")
    if not API_KEY:
        print(f"⚠️ PEXELS_API_KEY not set. Skipping Pexels downloads.")
        return 0
    
    downloaded = 0
    existing_hashes = set()
    
    # Get existing image hashes
    for img_file in save_dir.glob("*.jpg"):
        with open(img_file, "rb") as f:
            existing_hashes.add(get_image_hash(f.read()))
    
    try:
        url = "https://api.pexels.com/v1/search"
        params = {
            "query": monument_name,
            "per_page": min(count, 80)
        }
        headers = {"Authorization": API_KEY}
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"❌ Pexels API error: {response.status_code}")
            return 0
        
        data = response.json()
        
        for i, photo in enumerate(data.get("photos", [])):
            try:
                img_url = photo["src"]["large"]
                img_response = requests.get(img_url, timeout=30)
                
                if img_response.status_code == 200:
                    img_data = img_response.content
                    img_hash = get_image_hash(img_data)
                    
                    if img_hash not in existing_hashes:
                        filename = save_dir / f"pexels_{i}_{img_hash[:8]}.jpg"
                        with open(filename, "wb") as f:
                            f.write(img_data)
                        existing_hashes.add(img_hash)
                        downloaded += 1
                        print(f"  📥 Downloaded: {filename.name}")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error downloading image: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Pexels fetch error: {e}")
    
    return downloaded


def download_images_for_monument(monument_name):
    """Download images for a single monument from all sources."""
    folder_name = monument_name.replace(" ", "_").replace("/", "_")
    save_dir = DATASET_DIR / folder_name
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n🏛️ Fetching images for: {monument_name}")
    print(f"   Save directory: {save_dir}")
    
    total_downloaded = 0
    
    # Try each source
    total_downloaded += download_from_unsplash(monument_name, save_dir, 30)
    total_downloaded += download_from_pixabay(monument_name, save_dir, 30)
    total_downloaded += download_from_pexels(monument_name, save_dir, 30)
    
    # Count existing images
    existing_count = len(list(save_dir.glob("*.jpg"))) + len(list(save_dir.glob("*.png")))
    
    print(f"   ✅ Total images for {monument_name}: {existing_count}")
    
    return existing_count


def fetch_all_monuments():
    """Fetch images for all monuments."""
    print("=" * 60)
    print("🏛️ MONUMENT IMAGE FETCHER")
    print("=" * 60)
    
    # Create directories first
    create_directories()
    
    results = {}
    
    for monument in MONUMENTS:
        count = download_images_for_monument(monument)
        results[monument] = count
        time.sleep(1)  # Pause between monuments
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    
    total_images = 0
    for monument, count in results.items():
        status = "✅" if count >= 50 else "⚠️" if count >= 20 else "❌"
        print(f"{status} {monument}: {count} images")
        total_images += count
    
    print(f"\n📦 Total images collected: {total_images}")
    print(f"📁 Dataset location: {DATASET_DIR}")
    
    # Save metadata
    metadata = {
        "monuments": results,
        "total_images": total_images,
        "dataset_path": str(DATASET_DIR)
    }
    
    with open(BASE_DIR / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {BASE_DIR / 'dataset_metadata.json'}")


def manual_image_collector():
    """
    Instructions for manual image collection (Google Images, etc.)
    This is often needed to supplement API-collected images.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║           MANUAL IMAGE COLLECTION INSTRUCTIONS               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  For best results, supplement API images with manual         ║
║  collection from Google Images:                              ║
║                                                              ║
║  1. Use Chrome extension: "Download All Images"              ║
║  2. Search: "[Monument Name] HD photo"                       ║
║  3. Filter: Large images, Recent                             ║
║  4. Download to respective dataset folder                    ║
║                                                              ║
║  Alternative tools:                                          ║
║  - google-images-download (pip install)                      ║
║  - Bing Image Downloader                                     ║
║  - Flickr API                                                ║
║                                                              ║
║  Target: 100+ images per monument for good training          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monument Image Fetcher")
    parser.add_argument("--monument", type=str, help="Fetch images for specific monument")
    parser.add_argument("--create-dirs", action="store_true", help="Only create directories")
    parser.add_argument("--manual", action="store_true", help="Show manual collection instructions")
    
    args = parser.parse_args()
    
    if args.manual:
        manual_image_collector()
    elif args.create_dirs:
        create_directories()
    elif args.monument:
        download_images_for_monument(args.monument)
    else:
        fetch_all_monuments()
