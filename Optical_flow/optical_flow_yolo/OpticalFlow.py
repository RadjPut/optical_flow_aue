import cv2
import numpy as np


def harris_corners(image):
    count_harris = 0
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_img = np.float32(gray_img)

    corners_img = cv2.cornerHarris(gray_img, 3, 3, 0.04)
   # a = image[corners_img > 0]
   # print a.shape
    # Marking the corners in Green
    image[corners_img > 0.001 * corners_img.max()] = [0, 255, 0]

    for corners in corners_img:
        count_harris += 1

    print count_harris
    cv2.imwrite("harris_corners.png", image)

    return image


def shi_tomasi(image):
    count_tomasi = 0
    # Converting to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners
    corners_img = cv2.goodFeaturesToTrack(gray_img, 1000, 0.01, 10)

    corners_img = np.int0(corners_img)

    for corners in corners_img:
        x, y = corners.ravel()
        # Circling the corners in green
        cv2.circle(image, (x, y), 3, [0, 255, 0], -1)
        count_tomasi += 1

    cv2.imwrite("shi_timasi_features.png", image)
    cv2.imshow("shi_tomasi_features", image)
    cv2.waitKey(1)
    print (count_tomasi)
    return gray_img, corners_img, image


def good_features_2_track(frame):

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    a = detections[0]


    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    return p0, old_gray

def optical_flow(p0, old_gray, frame):

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    return p1, new_gray