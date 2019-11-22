from skimage.measure import compare_ssim
import imutils
import cv2

image_a = cv2.cvtColor(cv2.imread('2016.tif'), cv2.COLOR_BGR2GRAY)
x, y = image_a.shape
x = int(x / 2)
y = int(y / 2)
image_a = cv2.resize(image_a, (y, x)) / 65536
image_b = cv2.cvtColor(cv2.imread('2018.tif'), cv2.COLOR_BGR2GRAY)
image_b = cv2.resize(image_b, (y, x)) / 65536
(score, diff) = compare_ssim(image_a, image_b, full=True)
print(diff[0])
diff = (diff * 65536).astype('uint8')
print(diff[0])
thresh = cv2.threshold(diff, 0, 65536, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

image_a *= 65536
image_b *= 65536
image_a = cv2.cvtColor(image_a.astype('float32'), cv2.COLOR_GRAY2BGR)
image_b = cv2.cvtColor(image_b.astype('float32'), cv2.COLOR_GRAY2BGR)
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	if w * h < 128**2 or w * h > 512**2:
		continue
	cv2.rectangle(image_a, (int(x + w / 2), int(y + h / 2)), (int((x + w / 2) + 32), int((y + h / 2) + 32)), (255, 0, 0), cv2.FILLED)
	cv2.rectangle(image_b, (int(x + w / 2), int(y + h / 2)), (int((x + w / 2) + 32), int((y + h / 2) + 32)), (255, 0, 0), cv2.FILLED)

print("SSIM: {}".format(score))

cv2.imwrite('output_a.jpg', image_a)
cv2.imwrite('output_b.jpg', image_b)
cv2.imwrite('output.jpg', thresh)
