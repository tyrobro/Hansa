"""
Usage:
    python predict.py <image_path> [options]
    python predict.py <directory_path> --batch [options]

Options:
    --json              Output results in JSON format
    --visualize         Show image with prediction overlay
    --threshold FLOAT   Confidence threshold (default: 0.5)
    --batch            Process all images in directory
    --quiet            Minimal output
    --verbose          Detailed output
"""

import tensorflow as tf
import numpy as np
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ImageAuthenticityDetector:
    
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    DEFAULT_IMG_SIZE = 32
    
    def __init__(self, model_path: str = 'hansa.keras', img_size: int = DEFAULT_IMG_SIZE):
        
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.class_names = ['REAL', 'FAKE']
        
        self._load_config()
        
    def _load_config(self):
        config_path = 'training_config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.img_size = config.get('img_size', [self.DEFAULT_IMG_SIZE])[0]
                    self.class_names = config.get('class_names', self.class_names)
            except Exception:
                pass 
    
    def load_model(self):
        if self.model is not None:
            return
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file '{self.model_path}' not found. "
                "Please train the model first."
            )
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def _validate_image(self, image_path: str) -> bool:
        path = Path(image_path)
        return path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def load_and_prep_image(self, image_path: str) -> tf.Tensor:
        if not self._validate_image(image_path):
            raise ValueError(
                f"Unsupported image format. Supported formats: "
                f"{', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        try:
            img = tf.io.read_file(image_path)
            
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            
            img = tf.image.resize(img, [self.img_size, self.img_size])
            
            return img
            
        except Exception as e:
            raise ValueError(f"Failed to load image '{image_path}': {str(e)}")
    
    def predict(self, image_path: str, threshold: float = 0.5) -> Dict:
        
        self.load_model()
        
        img = self.load_and_prep_image(image_path)
        img_array = tf.expand_dims(img, axis=0)
        
        start_time = time.time()
        prediction = self.model.predict(img_array, verbose=0)
        inference_time = time.time() - start_time
        
        fake_score = float(prediction[0][0])
        real_score = 1 - fake_score
        
        predicted_class = 1 if fake_score > threshold else 0
        confidence = fake_score if predicted_class == 1 else real_score
        
        return {
            'image_path': image_path,
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'real_score': real_score,
            'fake_score': fake_score,
            'threshold': threshold,
            'inference_time_ms': inference_time * 1000
        }
    
    def predict_batch(self, directory_path: str, threshold: float = 0.5) -> List[Dict]:
        
        if not os.path.isdir(directory_path):
            raise ValueError(f"'{directory_path}' is not a valid directory")
        
        results = []
        image_files = []
        
        for file in Path(directory_path).iterdir():
            if file.is_file() and self._validate_image(str(file)):
                image_files.append(str(file))
        
        if not image_files:
            raise ValueError(f"No supported images found in '{directory_path}'")
        
        for image_file in image_files:
            try:
                result = self.predict(image_file, threshold)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_file,
                    'error': str(e)
                })
        
        return results


def print_result(result: Dict, verbose: bool = False):
    if 'error' in result:
        print(f"❌ Error processing {result['image_path']}: {result['error']}")
        return
    
    is_fake = result['predicted_class'] == 'FAKE'
    emoji = "⚠️" if is_fake else "✅"
    
    print(f"\n{emoji} {result['predicted_class']} IMAGE DETECTED")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    
    if verbose:
        print(f"   Real Score: {result['real_score']*100:.2f}%")
        print(f"   Fake Score: {result['fake_score']*100:.2f}%")
        print(f"   Threshold: {result['threshold']}")
        print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")
        print(f"   Image: {result['image_path']}")


def print_batch_summary(results: List[Dict]):
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    successful = total - errors
    
    if successful > 0:
        real_count = sum(1 for r in results if r.get('predicted_class') == 'REAL')
        fake_count = sum(1 for r in results if r.get('predicted_class') == 'FAKE')
        avg_confidence = np.mean([r['confidence'] for r in results if 'confidence' in r])
        
        print("\n" + "="*50)
        print("BATCH PREDICTION SUMMARY")
        print("="*50)
        print(f"Total Images: {total}")
        print(f"Successfully Processed: {successful}")
        print(f"Errors: {errors}")
        print(f"\n✅ Real Images: {real_count} ({real_count/successful*100:.1f}%)")
        print(f"⚠️  Fake Images: {fake_count} ({fake_count/successful*100:.1f}%)")
        print(f"\nAverage Confidence: {avg_confidence*100:.2f}%")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Hansa - AI Image Authenticity Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('path', help='Path to image file or directory')
    parser.add_argument('--json', action='store_true', 
                       help='Output in JSON format')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    parser.add_argument('--model', default='hansa.keras',
                       help='Path to model file (default: hansa.keras)')
    
    args = parser.parse_args()
    
    try:
        detector = ImageAuthenticityDetector(model_path=args.model)
        
        if args.batch:
            results = detector.predict_batch(args.path, args.threshold)
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                if not args.quiet:
                    for result in results:
                        print_result(result, args.verbose)
                print_batch_summary(results)
        else:
            result = detector.predict(args.path, args.threshold)
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_result(result, args.verbose)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image> [options]")
        print("Use --help for more options")
        sys.exit(1)
    
    main()
