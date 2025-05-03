import cv2
img = cv2.imread("cy.jpg")
resized_img = cv2.resize(img, (28,28))
cv2. imshow("cy", resized_img)
cv2.waitKey()
cv2.destroyAllWindows()

resized_img = resized_img[:,:,0]
cv2. imshow("cy", resized_img)
cv2.waitKey()
cv2.destroyAllWindows()
print(resized_img.shape)
