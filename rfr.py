import cv2 
import numpy as np
from skimage import img_as_float, img_as_ubyte

n2n = img_as_float(cv2.imread('./results_niid/N2N.jpg')[:, :, ::-1])
vdn = img_as_float(cv2.imread('./results_niid/VDN.jpg')[:, :, ::-1])
unet = img_as_float(cv2.imread('./results_niid/unet.jpg')[:, :, ::-1])

rfr = np.clip((n2n + vdn + unet) / 3 , 0, 1).astype(np.float32)

rfr = img_as_ubyte(rfr.clip(0,1))

cv2.imwrite('./results_niid/rfr.jpg', cv2.cvtColor(rfr, cv2.COLOR_BGR2RGB))