from PIL import Image
import cv2
import os
import PIL
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Rcv(object):


    # load image
    def load_image(self, path):
        new_image = Image.open(path)
        return new_image


    # create image
    def create_image(self, i, j):
        new_image = Image.new("RGB", (i, j), "black")
        return new_image


    # pixel value at row, column, channel
    def get_pixel(self, img, i, j, channel):
        if i < 0:
            i = 0
        if i > img.width:
            i = img.width - 1

        if j < 0:
            j = 0
        if j > img.height:
            j = img.height - 1

        if channel is 0:
            pixel = img.getpixel((i, j))[0]
        elif channel is 1:
            #  v = (pixels[i, j][0], value, pixels[i, j][2])
            pixel = img.getpixel((i, j))[1]
        elif channel is 2:
            pixel = img.getpixel((i, j))[2]
        return pixel


    # set pixel value; source, width, height, channel, value
    def set_pixel(self, img, i, j, c, value):
        pixels = img.load()
        # img =cv2.imread('./cats.jpg')
        if i < 0:
            i = 0
        if i > img.width:
            i = img.width - 1

        if j < 0:
            j = 0
        if j > img.height:
            j = img.height - 1

        if c is 0:
            pixels[i, j] = (value, pixels[i, j][1], pixels[i, j][2])
        elif c is 1:
            pixels[i, j] = (pixels[i, j][0], value, pixels[i, j][2])
        elif c is 2:
            pixels[i, j] = (pixels[i, j][0], pixels[i, j][1], value)


    def shift_image(self, img, i, j, c, value):
        pixels = img.load()
        # img =cv2.imread('./cats.jpg')
        #     if i < 0:
        #         i = 0
        #     if i > img.width:
        #         i = img.width-1

        #     if j < 0:
        #         j = 0
        #     if j > img.height:
        #         j = img.height-1

        if c is 0:
            pixels[i, j] = (value + pixels[i, j][0], pixels[i, j][1], pixels[i, j][2])
        elif c is 1:
            pixels[i, j] = (pixels[i, j][0], value + pixels[i, j][1], pixels[i, j][2])
        elif c is 2:
            pixels[i, j] = (pixels[i, j][0], pixels[i, j][1], value + pixels[i, j][2])


    def copy_image(self, image):
        new_image = Image.new("RGB", (image.width, image.height), "black")
        new_pixels = new_image.load()
        pixels = image.load()

        for i in range(image.width):
            for j in range(image.height):
                new_pixels[i, j] = (pixels[i, j][0], pixels[i, j][1], pixels[i, j][2])
        return new_pixels


    def save_image(self, file_name, image):
        #image = cv2.imread(image)
        cv2.imwrite(file_name, image)

    # def nn_interpolation(img, c):
    #     img = asarray(img)
    #     #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     width, height, channel = img.shape
    #
    #     a = int(width*c)
    #     b = int(height*c)
    #
    #     new_image = create_image(b, a)
    #     new_image = asarray(new_image)
    #     #new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    #
    #     #img = cv2.imread("./cats.jpg")
    #     #c += 4
    #     a = a - 1
    #     b = b - 1
    #
    #     for w in range(b):
    #         for e in range(a):
    #            # pixels[i, j] = (value, pixels[i, j][1], pixels[i, j][2])
    #             new_image[e, w] = (img[e//c, w//c], new_image[e, w][1], new_image[e, w][2])
    #             new_image[e, w] = (new_image[e, w][0], new_image[e, w][1], new_image[e, w][2])
    #             new_image[e, w] = (new_image[e, w][0], new_image[e, w][1], new_image[e, w][2])
    #             l = 0
    #
    #     return new_image

    def nn_interpolation(self, image, zoom, color_conversion_code):
        new_w = int(image.width*zoom)
        new_h = int(image.height*zoom)

        new_image = Image.new("RGB", (new_w, new_h), "black")
        new_pixels = new_image.load()
        pixels = image.load()

        for h in range(new_h-1):
            for w in range(new_w-1):
                new_pixels[w, h] = (pixels[w//zoom, h//zoom][0], pixels[w//zoom, h//zoom][1], pixels[w//zoom, h//zoom][2])

        new_image = asarray(new_image)
        new_image = cv2.cvtColor(new_image, color_conversion_code)

        return new_image

        # try:
        #     for h in range(height-1):
        #         for w in range(width-1):
        #             new_image[w*c, h*c] = img[w, h]
        # except Exception as ex:
        #     print(ex)
        # return  new_image
    def bl_interpolation(self, image, zoom, color_conversion_code):
        new_w = int(image.width*zoom)
        new_h = int(image.height*zoom)

 #       image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        new_image = Image.new("RGB", (new_w, new_h), "black")
      #  new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        new_pixels = new_image.load()

        #new_pixels = new_image.load()

        pixels = image.load()
        new_h = new_h - 1
        new_w = new_w - 1
        q11, q12, q21, q22 = pixels[0, 0], pixels[new_w//zoom, 0], pixels[0, new_h//zoom], pixels[new_w//zoom, new_h//zoom]
       # q11 = pixels[0, 0]
        #r = new_w//zoom
       # g,h = image.size
       # q12 = pixels[0, new_w // zoom]
        for h in range(new_h):
            for w in range(new_w):
                r1 = ((new_w//zoom - w//zoom)/(new_w//zoom - 0))*asarray(q11)+((w//zoom-0)/(new_w//zoom-0))*asarray(q21)
                r2 = ((new_w//zoom - w//zoom)/(new_w//zoom - 0))*asarray(q12)+((w//zoom-0)/(new_w//zoom-0))*asarray(q22)
                value = ((new_h//zoom-h//zoom)/(new_h//zoom-0))*asarray(r1)+((h//zoom-0)/(new_h//zoom-0))*asarray(r2)
                value = ([w // zoom, h // zoom][0], pixels[w // zoom, h // zoom][1], pixels[w // zoom, h // zoom][2])
                new_pixels[w//zoom, h//zoom] = value
                new_pixels[w, h] = value

        return new_image

#imgg = PIL.
