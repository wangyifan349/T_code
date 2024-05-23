import cv2
import numpy as np

# 读取图像
image = cv2.imread('image.jpg')

# 显示图像
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 显示灰度图
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 高斯模糊
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 显示模糊图像
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 边缘检测
edges = cv2.Canny(blurred_image, 50, 150)

# 显示边缘检测结果
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 膨胀图像
kernel = np.ones((5,5),np.uint8)
dilated_image = cv2.dilate(edges, kernel, iterations=1)

# 显示膨胀后的图像
cv2.imshow('Dilated Image', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 显示带有轮廓的图像
cv2.imshow('Image with Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 旋转图像
rows, cols = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

# 显示旋转后的图像
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 调整图像大小
resized_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# 显示调整大小后的图像
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 转换颜色空间（BGR到HSV）
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 显示HSV图像
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 提取颜色区域
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv_image, lower_red, upper_red)
color_extracted_image = cv2.bitwise_and(image, image, mask=mask)

# 显示提取颜色后的图像
cv2.imshow('Color Extracted Image', color_extracted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像阈值化
ret, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 显示阈值化图像
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像平滑（中值滤波）
median_blur_image = cv2.medianBlur(image, 5)

# 显示中值滤波后的图像
cv2.imshow('Median Blur Image', median_blur_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像修复
masked_image = cv2.imread('mask.jpg', 0)
inpainted_image = cv2.inpaint(image, masked_image, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# 显示修复后的图像
cv2.imshow('Inpainted Image', inpainted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像形态学操作（开运算）
kernel = np.ones((5,5),np.uint8)
opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 显示开运算后的图像
cv2.imshow('Opening Image', opening_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像轮廓逼近
epsilon = 0.1*cv2.arcLength(contour, True)
approx_contour = cv2.approxPolyDP(contour, epsilon, True)

# 绘制逼近后的轮廓
cv2.drawContours(image, [approx_contour], 0, (0, 255, 0), 2)

# 显示逼近后的图像
cv2.imshow('Approximated Contour Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像腐蚀
eroded_image = cv2.erode(image, kernel, iterations=1)

# 显示腐蚀后的图像
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像投影变换
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
perspective_matrix = cv2.getPerspectiveTransform(pts1,pts2)
perspective_transformed_image = cv2.warpPerspective(image, perspective_matrix, (300,300))

# 显示投影变换后的图像
cv2.imshow('Perspective Transformed Image', perspective_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像直方图均衡化
equalized_image = cv2.equalizeHist(gray_image)

# 显示均衡化后的图像
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
