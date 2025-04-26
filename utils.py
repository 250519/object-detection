import cv2
import numpy as np

def draw_bounding_box(frame, contour, label=None, color=(0, 255, 0)):
    """
    Draw a bounding box around a contour with optional label.
    
    Args:
        frame: Image to draw on
        contour: Contour to outline
        label: Optional text label to display
        color: RGB color tuple for the box
    """
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Draw the rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw the label if provided
    if label:
        y_label = y - 10 if y - 10 > 10 else y + h + 20
        cv2.putText(frame, label, (x, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def get_centroid(contour):
    """
    Calculate the centroid of a contour.
    
    Args:
        contour: Input contour
        
    Returns:
        tuple: (x, y) coordinates of centroid
    """
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        # Fallback to bounding box center if moments calculation fails
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w//2, y + h//2)

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (or None)
        height: Target height (or None)
        inter: Interpolation method
        
    Returns:
        Resized image
    """
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge channels back
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr