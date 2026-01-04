import numpy as np
from PIL import Image
from image_utils import detect_edges

def main():
    # 1. טעינת התמונה המקורית (וודאי שהקובץ נמצא באותה תיקייה)
    # החליפי את 'your_image.jpg' בשם הקובץ שיש לך
    img = Image.open('super_mario.jpg')
    img_array = np.array(img)
    
    # 2. הפעלת פונקציית זיהוי הקצוות
    edge_detected = detect_edges(img_array)
    
    # 3. המרה חזרה לתמונה ושמירה
    # מנרמלים את הערכים כדי שיוצגו נכון כתמונה
    edge_detected_rescaled = (edge_detected - edge_detected.min()) / (edge_detected.max() - edge_detected.min()) * 255
    output_image = Image.fromarray(edge_detected_rescaled.astype(np.uint8))
    
    output_image.save('edge_detected_result.png')
    print("Edge detection completed successfully!")

if __name__ == "__main__":
    main()

