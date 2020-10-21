from os import environ
from openalpr import Alpr
import cv2
import numpy as np
import json
import ctypes
import time


class Alpr_detection(object):

    def __init__(self, alpr_dir, country=None, config_file=None, runtime_dir=None):
        config_path = alpr_dir + "/openalpr.conf"
        runtime_path = alpr_dir + "/runtime_data"
        country_code = "eu"

        if config_file is not None:
            config_path = config_file
        if runtime_dir is not None:
            runtime_path = runtime_dir
        if country is not None:
            country_code = country

        environ["PATH"] = alpr_dir + ";" + environ["PATH"]
        self.__alpr = Alpr(country_code, config_path, runtime_path)

    def is_loaded(self):
        return self.__alpr.is_loaded()

    # configuration&alpr
    # def alpr_dir(self, alpr_dir):
    #     environ["PATH"] = alpr_dir + ";" + environ["PATH"]
    #     alpr = Alpr("eu", alpr_dir + "/openalpr.conf", alpr_dir + "/runtime_data")

    # plate frame
    # def alpr_detection(self):
    #     video = cv2.VideoCapture(0)  # catch video
    #     best_score = []
    #     k = 0
    #     while True:
    #         # catch video
    #         flag, frame = video.read()
    #         cv2.imshow('Clear_frame', frame)
    #         cv2.waitKey(2)
    #         if not flag:
    #             print('Video read failed.')
    #         # results = alpr.recognize_ndarray(frame)
    #         flag, enc = cv2.imencode("*.jpg", frame)  # ndarray issue
    #         #   results = alpr.recognize_array(bytes(bytearray(enc)))  # ndarray issue
    #         results = self.__alpr.recognize_array(enc.tobytes())

    def detection_photo(self, filePath):
        jpeg_bytes = open(filePath, "rb").read()
        return self.__alpr.recognize_array(jpeg_bytes)

    def detection_video(self, video, showCapFrame):
        # video = cv2.VideoCapture(0)  # catch video
        # catch video
        flag, frame = video.read()
        # width, height, c = frame.shape
        # frame = frame[450:width, 0:height]
        if showCapFrame is True:
            if frame is None:
                return
            cv2.imshow('Clear_frame', frame)
            cv2.waitKey(1)
        if not flag:
            print('Video read failed.')
            return None
        # results = alpr.recognize_ndarray(frame)

        flag, enc = cv2.imencode("*.jpg", frame)  # ndarray issue
        #   results = alpr.recognize_array(bytes(bytearray(enc)))  # ndarray issue
        results = self.__alpr.recognize_array(enc.tobytes())
        return results, frame
        # print results

        # return results, frame
        # print(results['results'])

    # def
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # converting BGR to HSV

    #  lower_red = np.array([30, 150, 50])  # define range of red color in HSV
    #  upper_red = np.array([255, 255, 180])

    # mask = cv2.inRange(hsv, lower_red, upper_red)  # create a red HSV colour boundary and threshold HSV image

    #  res = cv2.bitwise_and(frame, frame, mask=mask)  # Bitwise-AND mask and original image

    # cv2.imshow('Original', frame)  # Display an original image

    #  edges = cv2.Canny(frame, 100, 200)  # finds edges in the input image image and marks them in the output map edges

    # cv2.imshow('Original', edges)  # output edges

    # crop_frame = frame[0:480, 320:640]
    # cv2.imshow("Crop_frame", crop_frame)

    # crop_frame = frame[0:500, 0:640*2]
    # cv2.imshow("Crop_frame", crop_frame)
    def crop_frame(self, results, frame):

        if len(results['results']) == 0:
            z = 0  # result
        #  print('No plate detected')
        else:
            y, y1 = json.dumps(results["results"][0]["coordinates"][0]["y"]), json.dumps(
                results["results"][0]["coordinates"][3]["y"])
            x, x1 = json.dumps(results["results"][0]["coordinates"][0]["x"]), json.dumps(
                results["results"][0]["coordinates"][1]["x"])

            crop_frame = frame[int(y):int(y1), int(x):int(x1)]
            cv2.rectangle(frame, (int(x) - 10, int(y) - 10), (int(x1), int(y1)), (0, 255, 0), 3)
            # cv2.imshow("suka", crop_frame)
            # cv2.waitKey(2)
            return crop_frame

        # crop_frame_canny = cv2.Canny(crop_frame, 100, 200)
        # contours, hierarchy = cv2.findContours(crop_frame_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in contours:
        #     rect = cv2.minAreaRect(cnt)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     area = int(rect[1][0] * rect[1][1])
        #     if area > 500:
        #         cv2.drawContours(frame, [box], 0, (0, 255, 0), 3)
        # crop_frame = frame[int(x):int(x1), int(y):int(y1)]

    def print_results(self, frame, crop_frame, results):
        #  cv2.imshow("Crop_frame", frame)
        #  cv2.waitKey(2)
        if crop_frame is None:
            return

        # cv2.imshow("True frame", crop_frame)
        # cv2.waitKey(2)
        # cv2.imshow("Crop_frame", frame)
        # cv2.waitKey(2)
        for i, plate in enumerate(results['results']):
            best_candidate = plate['candidates'][0]
            if best_candidate['confidence'] > 85:
                #  for b in enumerate[best_score[b]]:
                best_score = best_candidate['confidence']
                # k += 1
                # if k == 2:
                #    print("plate: " + str(best_candidate['plate'].upper()))
                #    exit()

                print(best_candidate['confidence'])
                print("plate: " + str(best_candidate['plate'].upper()))
            # print('Plate #{}: {:7s} ({:.2f}%)'.format(i, best_candidate['plate'].upper(), best_candidate['confidence']))