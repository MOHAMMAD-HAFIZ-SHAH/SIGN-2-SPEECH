"""
Show what a training image looks like so you can match it with your webcam
"""
import cv2
import os

# Show sample training images
letters = ['A', 'B', 'C', 'E', 'M']

print("Displaying sample training images...")
print("Position your hand to match these images!")
print("\nPress any key to cycle through letters, ESC to exit\n")

for letter in letters:
    img_path = f'./dataset/{letter}/0.jpg'
    img = cv2.imread(img_path)
    
    if img is not None:
        # Resize for display
        display_img = cv2.resize(img, (640, 640))
        
        # Add text overlay
        cv2.putText(display_img, f"Letter: {letter}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(display_img, "Match your hand to this!", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow(f'Training Image Reference - Sign Language', display_img)
        key = cv2.waitKey(0)
        
        if key == 27:  # ESC key
            break

cv2.destroyAllWindows()
print("\nKey observations:")
print("- Hand should fill most of the frame")
print("- Clear, bright lighting")
print("- Plain or consistent background")
print("- Hand centered in frame")
