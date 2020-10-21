from classes import *
from OpticalFlow import harris_corners, shi_tomasi

yolo_detect = Detection(0)
video = cv2.VideoCapture(yolo_detect.getSource)
fist_iter = True
p0 = np.array([[[0, 0]]])
a = p0[0][0][0]
while True:
    ret, frame = video.read()
    boxes = yolo_detect.yoloDetection(frame)

    cv2.imshow("origin", frame)
    cv2.waitKey(1)
    # if len(boxes) != 0:
    # some = boxes[0]
    # frame = frame[some[1]:(some[1]+some[3]), some[0]:(some[0]+some[2])]

    if len(boxes) == 0:
        #cv2.destroyAllWindows()
        #video.release()
         #a = boxes[0][0]
        continue

   # if fist_iter:
   #     for detections in boxes:
   #
   #         ret, old_frame = video.read()
   #         # [some[1]:(some[1] + some[3]), some[0]:(some[0] + some[2])]
   #         p0, old_gray, mask = yolo_detect.shiTomasi(old_frame, detections)
   #         fist_iter = False

    #fist_iter = False
    # a = p0[0][0][0]
  #  if boxes is None:
  #      continue

    if (boxes[0][0] > p0[0][0][0] < boxes[0][1]) | (boxes[0][0]+boxes[0][2] < p0[0][0][0] > boxes[0][1]+boxes[0][3]):
        for detections in boxes:
            # if fist_iter:
            #ret, old_frame = video.read()

            p0, old_gray, mask = yolo_detect.shiTomasi(frame, detections)  # check p0 NaN values\111 Error nontype object iterable
            # cv2.imshow("asdasd", old_gray)

        # some = None
        # cv2.imshow("next frame", old_gray)
        # cv2.imshow("croped frame", new_old_gray)
        # cv2.waitKey(1)
    else:
        ret, new_frame = video.read()
        old_gray, p0, good_new, good_old = yolo_detect.opticalFlow(new_frame, p0, old_gray)
        yolo_detect.draw_optical_flow(mask, frame, good_new, good_old, boxes)
        frame = cv2.add(frame, mask)
        cv2.imshow("origin", frame)
        cv2.waitKey(1)

# flag1, frame1 = video.read()
# p0, old_gray, mask = yolo_detect.shiTomasi(frame1)
# while True:
#     flag, frame = video.read()
#
#     old_gray, p0, st = yolo_detect.opticalFlow(frame, p0, old_gray, mask)
#     if yolo_detect.opticalFlow(frame, p0, old_gray, mask) is None:
#         flag1, frame2 = video.read()
#         p0, old_gray, mask = yolo_detect.shiTomasi(frame2)
#    # yolo_detect.yoloDetection(frame)
#    # if yolo_detect.getObjectName == "Car":
#    #    yolo_detect.getResultTuple