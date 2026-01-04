from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינת התמונה מהנתיב שניתן והפיכתה למערך נמפיי
    img = Image.open(path)
    return np.array(img)

def edge_detection(image):
    # 1. הפיכת התמונה לגווני אפור (לפי שאלה 4 - מיצוע על ציר הצבעים)
    gray_image = np.mean(image, axis=2)
    
    # 2. הגדרת פילטר נגזרת בציר Y (לפי שאלה 3 שפתרנו)
    edge_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]])
    
    # 3. ביצוע הקונבולוציה (לפי שאלה 6 - שמירה על גודל וריפוד באפסים)
    edges = convolve2d(gray_image, edge_filter, mode='same', boundary='fill')
    
    return edges
