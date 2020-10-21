from classes import *
import cv2 as cv
Rcv = Rcv()
img = Rcv.load_image(r"C:\Users\r.pedan\ComputerVison\chess_desk.jpg")
#imgplot = plt.imshow(img)

#imgplot2 = plt.imshow(nn_interpolation(img, 2))
#plt.show()
img = Rcv.bl_interpolation(img, 2, "RGB")
#save_image("rtrtrtr.jpg", img)
#implot2 = plt.imshow(img)
#implot = plt.imshow(Rcv.nn_interpolation(img, 2, cv2.COLOR_BGR2GRAY))
#cv2.imshow("wsd", asarray(img))
#cv2.imshow("ws", Rcv.nn_interpolation(img, 4, cv2.COLOR_BGR2GRAY))
#cv2.imshow("wsq", asarray(Rcv.bl_interpolation(img, 4, cv2.COLOR_BGR2GRAY)))
#implot2 = plt.imshow(img)
#plt.show()


#res = cv.resize(asarray(img), (5*img.width, 5*img.height), interpolation=cv.INTER_CUBIC)
#resb = cv.resize(asarray(img), (5*img.width, 5*img.height), interpolation=cv.INTER_LINEAR)
cv2.imshow("a", img)
#cv2.imshow("ba", resb)
cv2.waitKey(999999)
