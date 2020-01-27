import numpy as np
import cv2
import os

# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def rotation(img_name, path_to_data):
    input_file = os.path.join(path_to_data,img_name+'.npy')
    #gt_file = os.path.join(path_to_data,'groundtruth',img_name+'.png')
    # HxWxC
    print("Loading img and gt...")
    img = np.load(input_file)
    #gt = cv2.imread(gt_file)

    #for i in range(1,4):

    rotated_img = np.rot90(img,3) # axis 1 by default
        #rotated_gt = rotate_bound(gt,360-i*90)

    print("Saving new img and gt rotated by")
    np.save(os.path.join(path_to_data, img_name +'test1'), rotated_img)
        #cv2.imwrite(os.path.join(path_to_data,'groundtruth', img_name + '_rotated_' + str(i * 90) + '.png'), rotated_gt)
    #print("Done.")

path_to_data = "/media/public_data/aeroscan/dataset/buildings_dataset/alldata/hues"
img_name = "outsiderotary_AeroscanKantoorVoorkantDak_2019-11-19_12-45-50"
rotation(img_name, path_to_data)