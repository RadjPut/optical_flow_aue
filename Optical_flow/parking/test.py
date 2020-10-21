from alpr import *
alpr_detection = Alpr_detection(r"C:\Users\r.pedan\GitLab\master\parksharing-cv\OpenALPR\openalpr_64")
video = cv2.VideoCapture("rtsp://cam:jhg23dfc@178.150.141.135:1555/Streaming/Channels/101")

while(True):

  #  ret, frame = video.read()
   # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results, frame = alpr_detection.detection_video(video, True)
    crop_frame = alpr_detection.crop_frame(results, frame)
    alpr_detection.print_results(frame, crop_frame, results)