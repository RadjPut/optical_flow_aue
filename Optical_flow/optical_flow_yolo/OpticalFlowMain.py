from OpticalFlow import harris_corners, shi_tomasi, good_features_2_track, optical_flow
import cv2

cap = cv2.VideoCapture(0)



#img = cv2.imread('car.png')
#cv2.imshow("Origin img", img)
#cv2.waitKey(9999)
#
#harris_corners = harris_corners(img)
while True:
    ret, fist_image = cap.read()
#    shi_tomasi_features = shi_tomasi(fist_image)
    p0, old_gray = good_features_2_track(fist_image)

    rett, new_frame = cap.read()

    p1, new_gray = optical_flow(p0, old_gray, new_frame)
