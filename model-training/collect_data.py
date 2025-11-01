"""
Data Collection Helper - Capture images from webcam for training
Optional tool to help collect additional training data
"""

import cv2
import os
import time
from datetime import datetime

class DataCollector:
    def __init__(self, output_dir='dataset', img_size=(200, 200)):
        self.output_dir = output_dir
        self.img_size = img_size
        self.current_sign = None
        self.count = 0
        
    def create_sign_directory(self, sign_name):
        """Create directory for a specific sign"""
        sign_dir = os.path.join(self.output_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        return sign_dir
    
    def collect_images(self, sign_name, num_images=100, delay=0.5):
        """
        Collect images for a specific sign
        
        Args:
            sign_name: Name of the sign (e.g., 'A', 'B', 'HELLO')
            num_images: Number of images to capture
            delay: Delay between captures in seconds
        """
        print(f"\n{'='*60}")
        print(f"Collecting images for sign: {sign_name}")
        print(f"Target: {num_images} images")
        print(f"{'='*60}\n")
        
        # Create directory
        sign_dir = self.create_sign_directory(sign_name)
        
        # Check existing images
        existing = len([f for f in os.listdir(sign_dir) if f.endswith('.jpg')])
        print(f"Existing images: {existing}")
        
        # Start from next number
        start_count = existing
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\nInstructions:")
        print("- Show the sign clearly to the camera")
        print("- Move your hand slightly between captures for variety")
        print("- Press 's' to start capturing")
        print("- Press 'q' to quit early")
        print("\nPress 's' to start...")
        
        capturing = False
        captured_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for preview
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Add text overlay
            if not capturing:
                cv2.putText(gray, "Press 's' to START", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                progress = f"Capturing: {captured_count}/{num_images}"
                cv2.putText(gray, progress, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(gray, f"Sign: {sign_name}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw rectangle for hand placement guide
            h, w = gray.shape
            center_x, center_y = w // 2, h // 2
            rect_size = 300
            cv2.rectangle(gray, 
                         (center_x - rect_size//2, center_y - rect_size//2),
                         (center_x + rect_size//2, center_y + rect_size//2),
                         (255, 255, 255), 2)
            
            cv2.imshow(f'Collecting: {sign_name}', gray)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Start capturing
            if key == ord('s') and not capturing:
                capturing = True
                print("\n✓ Capturing started!")
                time.sleep(1)  # Give time to prepare
            
            # Quit
            if key == ord('q'):
                print("\nCapture cancelled by user")
                break
            
            # Capture images
            if capturing and captured_count < num_images:
                # Extract ROI (region of interest)
                roi = gray[center_y - rect_size//2:center_y + rect_size//2,
                          center_x - rect_size//2:center_x + rect_size//2]
                
                # Resize to target size
                roi_resized = cv2.resize(roi, self.img_size)
                
                # Save image
                img_name = f"{sign_name}_{start_count + captured_count:04d}.jpg"
                img_path = os.path.join(sign_dir, img_name)
                cv2.imwrite(img_path, roi_resized)
                
                captured_count += 1
                print(f"  Captured: {captured_count}/{num_images}")
                
                # Delay between captures
                time.sleep(delay)
            
            # Done
            if captured_count >= num_images:
                print(f"\n✓ Successfully captured {captured_count} images!")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nImages saved to: {sign_dir}")
        print(f"Total images for '{sign_name}': {existing + captured_count}")


def main():
    """Interactive data collection"""
    print("="*60)
    print("Sign Language Data Collection Tool")
    print("="*60)
    
    # Configuration
    output_dir = input("\nEnter output directory (default: 'dataset'): ").strip() or 'dataset'
    
    collector = DataCollector(output_dir=output_dir)
    
    while True:
        print("\n" + "="*60)
        sign_name = input("Enter sign name (or 'quit' to exit): ").strip().upper()
        
        if sign_name.lower() in ['quit', 'q', 'exit']:
            print("\nExiting data collection")
            break
        
        if not sign_name:
            print("Please enter a valid sign name")
            continue
        
        try:
            num_images = int(input(f"Number of images to collect for '{sign_name}' (default: 100): ").strip() or '100')
            delay = float(input("Delay between captures in seconds (default: 0.5): ").strip() or '0.5')
        except ValueError:
            print("Invalid input, using defaults")
            num_images = 100
            delay = 0.5
        
        # Collect images
        collector.collect_images(sign_name, num_images, delay)
        
        another = input("\nCollect another sign? (y/n): ").strip().lower()
        if another != 'y':
            break
    
    print("\n" + "="*60)
    print("Data collection complete!")
    print("="*60)
    print(f"\nDataset saved in: {output_dir}")
    print("\nNext steps:")
    print("1. Review captured images and remove any poor quality ones")
    print("2. Run training script: python train_sign_model.py " + output_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nData collection interrupted")
        cv2.destroyAllWindows()
