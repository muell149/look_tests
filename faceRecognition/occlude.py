import cv2  
   
# path  
path = '../../CroppedYale/yaleB32/yaleB32_P00A-020E-10.pgm'
   
# Reading an image in default mode 
image = cv2.imread(path) 
   
# Window name in which image is displayed 
window_name = 'Image'
  
# Start coordinate, here (5, 5) 
# represents the top left corner of rectangle 
#start_point = (5, 15) 
start_point = (5, 70)

# Ending coordinate, here (220, 220) 
# represents the bottom right corner of rectangle 
#end_point = (160, 80) 
end_point = (160, 180)

# Blue color in BGR 
color = (0, 0, 0) 
  
# Line thickness of 2 px 
thickness = -1
  
# Using cv2.rectangle() method 
# Draw a rectangle with blue line borders of thickness of 2 px 
image = cv2.rectangle(image, start_point, end_point, color, thickness) 

cv2.imwrite('occludetest2.pgm', image)

# Displaying the image  
cv2.imshow(window_name, image)
cv2.waitKey(0)