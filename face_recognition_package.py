
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFaceRecognition:
    def __init__(self):
        """Initialize the face recognition system"""
        logger.info("Face Recognition System initialized")
    
    def convert_image_format(self, image_path):
        """
        Convert image to RGB format that face_recognition can handle
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image in RGB format or None if failed
        """
        try:
            # Open image with PIL
            img = Image.open(image_path)
            
            # Get original mode for logging
            original_mode = img.mode
            logger.info(f"Original image mode: {original_mode}")
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    # Handle transparency by creating white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                    logger.info("Converted RGBA to RGB with white background")
                elif img.mode == 'CMYK':
                    # Convert CMYK to RGB
                    img = img.convert('RGB')
                    logger.info("Converted CMYK to RGB")
                elif img.mode == 'L':
                    # Convert grayscale to RGB
                    img = img.convert('RGB')
                    logger.info("Converted grayscale to RGB")
                elif img.mode == 'P':
                    # Convert palette to RGB
                    img = img.convert('RGB')
                    logger.info("Converted palette to RGB")
                else:
                    # Try generic conversion to RGB
                    img = img.convert('RGB')
                    logger.info(f"Converted {original_mode} to RGB")
            
            # Ensure 8-bit depth
            if img.mode == 'RGB':
                # Convert to numpy array and ensure 8-bit
                img_array = np.array(img)
                if img_array.dtype != np.uint8:
                    # Normalize to 8-bit if needed
                    if img_array.max() > 255:
                        img_array = (img_array / img_array.max() * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                    img = Image.fromarray(img_array)
                    logger.info("Converted to 8-bit depth")
            
            return img
            
        except Exception as e:
            logger.error(f"Error converting image format: {e}")
            return None
        
    def load_and_analyze_image(self, image_path):
        """
        Load image and detect faces with encodings
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with face data
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
            
        try:
            # Method 1: Try using face_recognition.load_image_file directly
            try:
                # First convert image to proper format and save temporarily if needed
                pil_image = self.convert_image_format(image_path)
                if pil_image is None:
                    logger.error(f"Failed to convert image format: {image_path}")
                    return None
                
                # Try direct loading first
                image = face_recognition.load_image_file(image_path)
                logger.info(f"Successfully loaded with face_recognition.load_image_file: {image.shape}, dtype: {image.dtype}")
                
            except Exception as e:
                logger.warning(f"Direct loading failed: {e}")
                logger.info("Trying alternative method...")
                
                # Method 2: Convert through PIL and ensure proper format
                # Convert PIL image to numpy array for face_recognition
                image = np.array(pil_image)
                
                # Ensure the array is contiguous and in the right format
                if not image.flags['C_CONTIGUOUS']:
                    image = np.ascontiguousarray(image)
                    logger.info("Made array contiguous")
                
                # Verify image format
                if len(image.shape) != 3 or image.shape[2] != 3:
                    logger.error(f"Image must be RGB format, got shape: {image.shape}")
                    return None
                    
                if image.dtype != np.uint8:
                    logger.error(f"Image must be 8-bit, got dtype: {image.dtype}")
                    return None
                
                logger.info(f"Image processed successfully: {image.shape}, dtype: {image.dtype}")
            
            # Find face locations (bounding boxes)
            face_locations = face_recognition.face_locations(image)
            logger.info(f"Found {len(face_locations)} face(s)")
            
            # Get face encodings (facial features)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            return {
                'image': image,
                'pil_image': pil_image if 'pil_image' in locals() else Image.fromarray(image),  # Keep PIL version for display
                'face_locations': face_locations,
                'face_encodings': face_encodings,
                'face_count': len(face_locations)
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def compare_faces(self, image_path1, image_path2, tolerance=0.4):
        """
        Compare faces between two images
        
        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            tolerance: Lower values make face comparison more strict (default: 0.4)
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing faces: {os.path.basename(image_path1)} vs {os.path.basename(image_path2)}")
        
        # Analyze both images
        data1 = self.load_and_analyze_image(image_path1)
        data2 = self.load_and_analyze_image(image_path2)
        
        if not data1 or not data2:
            return None
            
        # Initialize results
        results = {
            'image1_path': image_path1,
            'image2_path': image_path2,
            'faces1': data1['face_locations'],
            'faces2': data2['face_locations'],
            'face_count1': data1['face_count'],
            'face_count2': data2['face_count'],
            'match': False,
            'confidence': 0.0,
            'distance': 1.0,
            'tolerance': tolerance,
            'all_comparisons': []
        }
        
        # Store images for display
        results['image1'] = data1['pil_image']
        results['image2'] = data2['pil_image']
        
        # Compare faces if both images have faces
        if len(data1['face_encodings']) > 0 and len(data2['face_encodings']) > 0:
            
            # Compare each face in image1 with each face in image2
            for i, encoding1 in enumerate(data1['face_encodings']):
                for j, encoding2 in enumerate(data2['face_encodings']):
                    
                    # Calculate face distance (lower = more similar)
                    distance = face_recognition.face_distance([encoding1], encoding2)[0]
                    
                    # Check if faces match
                    match = distance < tolerance
                    
                    # Convert distance to confidence percentage
                    confidence = (1 - distance) * 100
                    
                    comparison = {
                        'face1_index': i,
                        'face2_index': j,
                        'distance': distance,
                        'confidence': confidence,
                        'match': match
                    }
                    
                    results['all_comparisons'].append(comparison)
            
            # Find best match
            if results['all_comparisons']:
                best_match = min(results['all_comparisons'], key=lambda x: x['distance'])
                results['match'] = best_match['match']
                results['confidence'] = best_match['confidence']
                results['distance'] = best_match['distance']
                results['best_match'] = best_match
        
        return results
    
    def display_results(self, results):
        """Display comparison results with face bounding boxes"""
        if not results:
            print("No results to display")
            return
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Display images
        ax1.imshow(results['image1'])
        ax1.set_title(f'Image 1 - {results["face_count1"]} face(s) detected')
        ax1.axis('off')
        
        ax2.imshow(results['image2'])
        ax2.set_title(f'Image 2 - {results["face_count2"]} face(s) detected')
        ax2.axis('off')
        
        # Draw bounding boxes for image 1
        for i, (top, right, bottom, left) in enumerate(results['faces1']):
            bbox = patches.Rectangle(
                (left, top), right - left, bottom - top,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax1.add_patch(bbox)
            ax1.text(left, top - 10, f"Face {i+1}", 
                   bbox=dict(facecolor='red', alpha=0.7), 
                   fontsize=9, color='white')
        
        # Draw bounding boxes for image 2
        for i, (top, right, bottom, left) in enumerate(results['faces2']):
            bbox = patches.Rectangle(
                (left, top), right - left, bottom - top,
                linewidth=2, edgecolor='blue', facecolor='none'
            )
            ax2.add_patch(bbox)
            ax2.text(left, top - 10, f"Face {i+1}", 
                   bbox=dict(facecolor='blue', alpha=0.7), 
                   fontsize=9, color='white')
        
        # Add main title with results
        match_text = "✓ FACES MATCH" if results['match'] else "✗ FACES DON'T MATCH"
        color = 'green' if results['match'] else 'red'
        
        title = f"Face Recognition Comparison\n"
        title += f"Confidence: {results['confidence']:.1f}% | Distance: {results['distance']:.3f} | {match_text}"
        
        fig.suptitle(title, fontsize=14, fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        self.print_detailed_results(results)
    
    def print_detailed_results(self, results):
        """Print detailed comparison results"""
        print(f"\n{'='*60}")
        print("FACE RECOGNITION COMPARISON RESULTS")
        print(f"{'='*60}")
        
        print(f"Image 1: {os.path.basename(results['image1_path'])}")
        print(f"Image 2: {os.path.basename(results['image2_path'])}")
        print(f"Faces detected: {results['face_count1']} + {results['face_count2']}")
        
        print(f"\nBest Match:")
        print(f"  Match: {results['match']}")
        print(f"  Confidence: {results['confidence']:.1f}%")
        print(f"  Distance: {results['distance']:.3f}")
        print(f"  Tolerance: {results['tolerance']}")
        
        # Print all comparisons if multiple faces
        if len(results['all_comparisons']) > 1:
            print(f"\nAll Face Comparisons:")
            for i, comp in enumerate(results['all_comparisons']):
                print(f"  Face {comp['face1_index']+1} vs Face {comp['face2_index']+1}: "
                      f"{comp['confidence']:.1f}% confidence, "
                      f"{'MATCH' if comp['match'] else 'NO MATCH'}")
        
        print(f"\nInterpretation:")
        if results['confidence'] > 80:
            print("  🟢 Very high confidence - faces are very likely the same person")
        elif results['confidence'] > 60:
            print("  🟡 High confidence - faces are likely the same person")
        elif results['confidence'] > 40:
            print("  🟠 Moderate confidence - faces might be the same person")
        else:
            print("  🔴 Low confidence - faces are likely different people")

    def debug_face_recognition_compatibility(self, image_path):
        """Debug face_recognition library compatibility"""
        print(f"\n🔧 DEBUGGING: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return False
            
        try:
            # Test 1: PIL Image loading
            print("Test 1: PIL Image loading...")
            pil_img = Image.open(image_path)
            print(f"✅ PIL: {pil_img.mode}, {pil_img.size}, {pil_img.format}")
            
            # Test 2: Convert to RGB
            print("Test 2: Converting to RGB...")
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            print(f"✅ RGB: {pil_img.mode}, {pil_img.size}")
            
            # Test 3: Convert to numpy
            print("Test 3: Converting to numpy...")
            np_img = np.array(pil_img)
            print(f"✅ NumPy: {np_img.shape}, {np_img.dtype}")
            
            # Test 4: Make contiguous
            print("Test 4: Making contiguous...")
            if not np_img.flags['C_CONTIGUOUS']:
                np_img = np.ascontiguousarray(np_img)
                print(f"✅ Made contiguous: {np_img.flags['C_CONTIGUOUS']}")
            else:
                print(f"✅ Already contiguous: {np_img.flags['C_CONTIGUOUS']}")
            
            # Test 5: Direct face_recognition loading
            print("Test 5: Direct face_recognition loading...")
            try:
                fr_img = face_recognition.load_image_file(image_path)
                print(f"✅ Direct loading: {fr_img.shape}, {fr_img.dtype}")
                
                # Test face detection on direct load
                print("Test 5a: Face detection on direct load...")
                locations = face_recognition.face_locations(fr_img)
                print(f"✅ Found {len(locations)} faces with direct load")
                return True
                
            except Exception as e:
                print(f"❌ Direct loading failed: {e}")
            
            # Test 6: Face detection on converted image
            print("Test 6: Face detection on converted image...")
            try:
                locations = face_recognition.face_locations(np_img)
                print(f"✅ Found {len(locations)} faces with converted image")
                return True
            except Exception as e:
                print(f"❌ Face detection failed: {e}")
                
            # Test 7: Try different image formats
            print("Test 7: Trying different formats...")
            
            # Save as different formats and try
            import tempfile
            for fmt in ['PNG', 'JPEG']:
                try:
                    with tempfile.NamedTemporaryFile(suffix=f'.{fmt.lower()}', delete=False) as tmp:
                        pil_img.save(tmp.name, format=fmt)
                        tmp_img = face_recognition.load_image_file(tmp.name)
                        locations = face_recognition.face_locations(tmp_img)
                        print(f"✅ {fmt} format: Found {len(locations)} faces")
                        os.unlink(tmp.name)
                        return True
                except Exception as e:
                    print(f"❌ {fmt} format failed: {e}")
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass
                        
            return False
                
        except Exception as e:
            print(f"❌ Debug failed: {e}")
            return False

def main():
    """Example usage with diagnostics"""
    # Initialize face recognition
    face_rec = SimpleFaceRecognition()
    
    # Image paths - UPDATE THESE WITH YOUR ACTUAL IMAGE PATHS
    image_path1 = r"C:\Users\Olumayowa.Oyaleke\Downloads\Image (7).jpg"
    image_path2 = r"C:\Users\Olumayowa.Oyaleke\Downloads\Image (8).jpg"
    
    # Check if files exist
    if not os.path.exists(image_path1):
        print(f"❌ Image 1 not found: {image_path1}")
        print("Please update image_path1 with the correct path to your first image")
        return
        
    if not os.path.exists(image_path2):
        print(f"❌ Image 2 not found: {image_path2}")
        print("Please update image_path2 with the correct path to your second image")
        return
    
    # Diagnose both images first
    print("🔍 DEBUGGING IMAGES...")
    debug1 = face_rec.debug_face_recognition_compatibility(image_path1)
    debug2 = face_rec.debug_face_recognition_compatibility(image_path2)
    
    if not debug1 or not debug2:
        print("\n❌ Debugging revealed compatibility issues.")
        print("Try the following:")
        print("1. Re-save images in a different format (PNG/JPEG)")
        print("2. Check if the images are corrupted")
        print("3. Try smaller image sizes")
        print("4. Make sure the images actually contain faces")
        return
    
    print("\n" + "="*50)
    print("🚀 Starting face recognition comparison...")
    print("📸 Processing images locally - no data sent to external servers")
    
    # Compare faces
    results = face_rec.compare_faces(image_path1, image_path2)
    
    if results:
        # Display results
        face_rec.display_results(results)
    else:
        print("❌ Failed to process images. Check the diagnostics above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure images are valid image files")
        print("2. Try saving images in a different format (PNG or JPG)")
        print("3. Check if images are corrupted")
        print("4. Ensure images contain faces")

def quick_test():
    """Quick test with different tolerance levels"""
    face_rec = SimpleFaceRecognition()
    
    # Your image paths
    image_path1 = r"C:\Users\Olumayowa.Oyaleke\Downloads\Image (6).jpg"
    image_path2 = r"C:\Users\Olumayowa.Oyaleke\Downloads\output-onlinetools.png"
    
    # Test different tolerance levels
    tolerances = [0.3, 0.4, 0.5, 0.6]
    
    print("Testing different tolerance levels:")
    print("(Lower tolerance = more strict matching)")
    
    for tolerance in tolerances:
        results = face_rec.compare_faces(image_path1, image_path2, tolerance=tolerance)
        if results:
            match_status = "MATCH" if results['match'] else "NO MATCH"
            print(f"Tolerance {tolerance}: {results['confidence']:.1f}% confidence - {match_status}")

if __name__ == "__main__":
    print("Simple Face Recognition System - FIXED VERSION")
    print("=" * 50)
    print("✅ Secure local processing")
    print("✅ No external API calls")
    print("✅ Complete privacy")
    print("✅ Handles various image formats")
    print("✅ Built-in diagnostics")
    print()
    print("Required: pip install face-recognition matplotlib pillow")
    print()
    
    # Run main comparison with diagnostics
    main()
    
    # Uncomment to test different tolerance levels
    # print("\n" + "="*50)
    # quick_test()



# image = face_recognition.load_image_file(r"C:\Users\Olumayowa.Oyaleke\Downloads\Image(6).png")
# face_landmarks_list = face_recognition.face_landmarks(image)