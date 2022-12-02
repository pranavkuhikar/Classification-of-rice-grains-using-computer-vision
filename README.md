# Classification-of-rice-grains-using-computer-vision
The aim of this project is to count the total number of grains of rice and classify them according to their types.
Any image of rice grains from google can be used for this task
We need to classify the rice particle according to their types, i.e. slender, medium, bold, round, dust.
This can be done by alloting the size of the ratio
# classification of rice particles

# classification(ratio):
    (ratio>=3 and ratio<3.5): slender
    (ratio>=2.1 and ratio<3): "Medium"
    (ratio>=1.1 and ratio<2.1): "Bold"
    (ratio>=0.9 and ratio<=1): "Round"
    otherwise "Dust"
    
Now we need to filter and specify the threshold, kernel for identifying the grains 
# convert to binary
# 160 - threhold, 255 - value to assign, thresh_binary_inv - Inverse binary
ret, binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)

# avg filter
kernel = np.ones((5,5),np.float32)/9
dst = cv2.filter2D(binary,-1,kernel)

# -1: Depth of the destination image
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# erosion
erosion = cv2.erode(dst, kernel2, iterations =1)

# dilation
dilation = cv2.dilate(erosion, kernel2, iterations =1)

# edge detection 
edges = cv2.Canny(dilation,100,200)

# size detection - This will help us count the no. of grains.
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("No. of rice grains=", len(contours))
total_ar=0

By using the aspect ratio we an identify the impurities.
then we need to use the values of ratio to classify the grains.
Print the no.of grains.



