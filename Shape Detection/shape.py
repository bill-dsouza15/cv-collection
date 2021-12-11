# Shape Detection
# 
# This module contains the Shape class which returns the shape of the object
# based on approximations made on contours

import cv2

# Driver function
class Shape:
    def __init__(self):
        pass
    
    def getShape(self, contour):
        # Approximate the shape using contours
        n_points = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        
        # If n_points = 3 then it is a triangle
        if len(n_points) == 3:
            return "Triangle"
        
        # If n_points = 4 then it is a quadrilateral
        elif len(n_points) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 0.95 * h and w <= 1.05 * h:
                return "Square"
            else:
                return "Rectangle"

        # If n_points = 5 then it is a pentagon
        elif len(n_points) == 5:
            return "Pentagon"

        # If n_points = 6 then it is a hexagon
        elif len(n_points) == 6:
            return "Hexagon"
            
        # Otherwise it is assumed to be a circle
        else:
            return "Circle"
