from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageStat, ImageFilter
import cv2 as cv
#import image_slicer
import argparse
import imutils


# Load the image
sova_image = cv.imread('car.png')

# Convert the image to RGB
sova_image = cv.cvtColor(sova_image, cv.COLOR_BGR2RGB)
# Convert the image to grayscale
sova_image_gray = cv.cvtColor(sova_image, cv.COLOR_RGB2GRAY)

cv.imshow("Original image", sova_image)

cv.waitKey(10000)

################ 1 ORB BRIK KEYPOINTS DETECT ########################################

orb = cv.ORB_create()
sova_keypoints_orb, sova_descriptor_orb = orb.detectAndCompute(sova_image_gray, None)
orb_keypoints_sova_image = np.copy(sova_image)
cv.drawKeypoints(sova_image, sova_keypoints_orb, orb_keypoints_sova_image, color = (0,  255, 0))

##################################################################################################

brisk = cv.BRISK_create()
sova_keypoints_brisk, sova_descriptor_brisk = brisk.detectAndCompute(sova_image_gray, None)
brisk_keypoints_sova_image = np.copy(sova_image)
cv.drawKeypoints(sova_image, sova_keypoints_brisk, brisk_keypoints_sova_image, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (255, 36 , 0))


# Display image ORB BRISK keypoints

cv.imshow("ORB keypoints detected: %s" % len(sova_keypoints_orb), orb_keypoints_sova_image)
cv.imshow("BRISK keypoints detected: %s" % len(sova_keypoints_brisk), brisk_keypoints_sova_image)

cv.waitKey(100000)


################ 2 JOINT ACTION MATRIX ########################################

matrix_orb = []
k = 0
for r in range(17):
    matrix_orb.append([])
    for c in range(17):
        matrix_orb[r].append(k)

matrix_brisk = []
k = 0
for r in range(17):
    matrix_brisk.append([])
    for c in range(17):
        matrix_brisk[r].append(k)

matrix_orb_brisk = []
k = 0
for r in range(17):
    matrix_orb_brisk.append([])
    for c in range(17):
        matrix_orb_brisk[r].append(k)

# Slicer
Ox, Oy, Ox1, Oy1 = -25, -25, 0, 0
count = 0
hit_orb = 0  # hit counter
hit_brisk = 0  # hit train counter
#i, j = 0, 0

# width, height = original.size
for j in range(0, 16):
    Ox = 0  # left
    Ox1 = 25  # right
    Oy += 25  # upper
    Oy1 += 25  # lower

    for i in range(0, 16):
        Ox = 25 + Ox
        Ox1 = 25 + Ox1
        hit = 0
        # hit counter
        # coordinates of feature point compare to slice coordinates
        for k in range(0, len(sova_keypoints_orb)):
            if (Ox <= sova_keypoints_orb[k].pt[0] <= Ox1) & (Oy <= sova_keypoints_orb[k].pt[1] <= Oy1):
                hit_orb += 1
                matrix_orb[j][i] = hit_orb
        for s in range(0, len(sova_keypoints_brisk)):
            if (Ox <= sova_keypoints_brisk[s].pt[0] <= Ox1) & (Oy <= sova_keypoints_brisk[s].pt[1] <= Oy1):
                hit_orb += 1
                matrix_brisk[j][i] = hit_orb

for v in range(17):
    for r in range(17):
            matrix_orb_brisk[v][r] = abs(matrix_orb[v][r] & matrix_brisk[v][r])

# Print results
print("orb:")
for n in range(0, len(matrix_orb)):
    print(matrix_orb[n])
print("brisk:")

for n in range(0, len(matrix_brisk)):
    print(matrix_brisk[n])
print("orb and brisk:")

for n in range(0, len(matrix_orb_brisk)):
    print(matrix_orb_brisk[n])
print()

################ 3 Rotate image ########################################

rotated_sova = imutils.rotate_bound(sova_image, 30)
cv.imshow("Rotated (Correct)", rotated_sova)
cv.waitKey(3000)

# Convert the image to RGB
rotated_sova = cv.cvtColor(rotated_sova, cv.COLOR_BGR2RGB)
# Convert the image to grayscale
rotated_sova_gray = cv.cvtColor(rotated_sova, cv.COLOR_RGB2GRAY)

# orb brisk keypoints
orb = cv.ORB_create()
rotated_sova_keypoints_orb, rotated_sova_descriptor_orb = orb.detectAndCompute(rotated_sova_gray, None)
orb_keypoints_rotated_sova_image = np.copy(rotated_sova)
cv.drawKeypoints(rotated_sova, rotated_sova_keypoints_orb, orb_keypoints_rotated_sova_image, color = (0,  255, 0))

##################################################################################################

brisk = cv.BRISK_create()
rotated_sova_keypoints_brisk, rotated_sova_descriptor_brisk = brisk.detectAndCompute(rotated_sova_gray, None)
brisk_keypoints_rotated_sova_image = np.copy(rotated_sova)
cv.drawKeypoints(rotated_sova, rotated_sova_keypoints_brisk, brisk_keypoints_rotated_sova_image,flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (255, 36 , 0))


# Display image ORB BRISK keypoints

cv.imshow("ORB keypoints detected: %s" % len(rotated_sova_keypoints_orb), orb_keypoints_rotated_sova_image)

cv.imshow("BRISK keypoints detected: %s" % len(rotated_sova_keypoints_brisk), brisk_keypoints_rotated_sova_image)

cv.waitKey(100000)

matrix_orb = []
k = 0
for r in range(17):
    matrix_orb.append([])
    for c in range(17):
        matrix_orb[r].append(k)

matrix_brisk = []
k = 0
for r in range(17):
    matrix_brisk.append([])
    for c in range(17):
        matrix_brisk[r].append(k)

matrix_orb_brisk = []
k = 0
for r in range(17):
    matrix_orb_brisk.append([])
    for c in range(17):
        matrix_orb_brisk[r].append(k)


# Slicer
Ox, Oy, Ox1, Oy1 = -25, -25, 0, 0
count = 0
hit_orb = 0  # hit counter
hit_brisk = 0  # hit train counter
#i, j = 0, 0

# width, height = original.size
for j in range(0, 16):
    Ox = 0  # left
    Ox1 = 25  # right
    Oy += 25  # upper
    Oy1 += 25  # lower

    for i in range(0, 16):
        Ox = 25 + Ox
        Ox1 = 25 + Ox1
        hit = 0
        # hit counter
        # coordinates of feature point compare to slice coordinates
        for k in range(0, len(rotated_sova_keypoints_orb)):
            if (Ox <= rotated_sova_keypoints_orb[k].pt[0] <= Ox1) & (Oy <= rotated_sova_keypoints_orb[k].pt[1] <= Oy1):
                hit_orb += 1
                matrix_orb[j][i] = hit_orb
        for s in range(0, len(rotated_sova_keypoints_brisk)):
            if (Ox <= rotated_sova_keypoints_brisk[s].pt[0] <= Ox1) & (Oy <= rotated_sova_keypoints_brisk[s].pt[1] <= Oy1):
                hit_orb += 1
                matrix_brisk[j][i] = hit_orb

for v in range(17):
    for r in range(17):
            matrix_orb_brisk[v][r] = abs(matrix_orb[v][r] & matrix_brisk[v][r])

# Print results
print("orb:")
for n in range(0, len(matrix_orb)):
    print(matrix_orb[n])

print("brisk:")

for n in range(0, len(matrix_brisk)):
    print(matrix_brisk[n])

print("orb and brisk:")

for n in range(0, len(matrix_orb_brisk)):
    print(matrix_orb_brisk[n])


################ 4 SCALE ########################################

#cv.imshow('Scaling - increase', sova_image)
img_increased = cv.resize(sova_image,(638, 638), interpolation = cv.INTER_CUBIC)
#cv.imshow("img_increased", img_increased)

#cv.waitKey(1000)
#cv2.imshow("Rotated (Problematic)", image)
#cv.waitKey(1000)


#cv.imshow('Scaling - decrease', sova_image)
img_decreased = cv.resize(sova_image,(238, 238), interpolation = cv.INTER_AREA)
#cv.imshow("img_decreased", img_decreased)

#cv.waitKey(1000)
#cv2.imshow("Rotated (Problematic)", image)
#cv.waitKey(10000000)

# Convert the image to RGB
img_increased = cv.cvtColor(img_increased, cv.COLOR_BGR2RGB)
img_decreased = cv.cvtColor(img_decreased, cv.COLOR_BGR2RGB)
# Convert the image to grayscale
img_increased_gray = cv.cvtColor(img_increased, cv.COLOR_RGB2GRAY)
img_decreased_gray = cv.cvtColor(img_decreased, cv.COLOR_RGB2GRAY)


# orb brisk keypoints
orb = cv.ORB_create()
img_increased_keypoints_orb, img_increased_descriptor_orb = orb.detectAndCompute(img_increased_gray, None)
img_decreased_keypoints_orb, img_decreased_descriptor_orb = orb.detectAndCompute(img_decreased_gray, None)

orb_keypoints_img_increased_image = np.copy(img_increased)
orb_keypoints_img_decreased_image = np.copy(img_decreased)

#cv.drawKeypoints(img_increased, img_increased_keypoints_orb, orb_keypoints_img_increased_image, color = (0,  255, 0))
#cv.drawKeypoints(img_decreased, img_decreased_keypoints_orb, orb_keypoints_img_decreased_image, color = (0,  255, 0))


##################################################################################################

brisk = cv.BRISK_create()
img_increased_keypoints_brisk, img_increased_descriptor_brisk = brisk.detectAndCompute(img_increased_gray, None)
img_decreased_keypoints_brisk, img_decreased_descriptor_brisk = brisk.detectAndCompute(img_decreased_gray, None)

brisk_keypoints_img_increased_image = np.copy(img_increased)
brisk_keypoints_img_decreased_image = np.copy(img_decreased)

#cv.drawKeypoints(img_increased, img_increased_keypoints_brisk, brisk_keypoints_img_increased_image, color = (255,  36, 0))
#cv.drawKeypoints(img_decreased, img_decreased_keypoints_brisk, brisk_keypoints_img_decreased_image, color = (255,  36, 0))


# Display image ORB BRISK keypoints

#cv.imshow("ORB keypoints detected incr: %s" % len(img_increased_keypoints_orb), orb_keypoints_img_increased_image)
#cv.imshow("ORB keypoints detected decr: %s" % len(img_decreased_keypoints_orb), orb_keypoints_img_decreased_image)

#cv.imshow("BRISK keypoints detected incr: %s" % len(img_increased_keypoints_brisk), brisk_keypoints_img_increased_image)
#cv.imshow("BRISK keypoints detected decr: %s" % len(img_decreased_keypoints_brisk), brisk_keypoints_img_decreased_image)

#cv.waitKey(12000)


#
# matrix_orb = []
# k = 0
# for r in range(17):
#     matrix_orb.append([])
#     for c in range(17):
#         matrix_orb[r].append(k)
#
# matrix_brisk = []
# k = 0
# for r in range(17):
#     matrix_brisk.append([])
#     for c in range(17):
#         matrix_brisk[r].append(k)
#
# matrix_orb_brisk = []
# k = 0
# for r in range(17):
#     matrix_orb_brisk.append([])
#     for c in range(17):
#         matrix_orb_brisk[r].append(k)


# Slicer
Ox, Oy, Ox1, Oy1 = -25, -25, 0, 0
count = 0
hit_orb = 0  # hit counter
hit_brisk = 0  # hit train counter
#i, j = 0, 0

# width, height = original.size
for j in range(0, 16):
    Ox = 0  # left
    Ox1 = 25  # right
    Oy += 25  # upper
    Oy1 += 25  # lower

    for i in range(0, 16):
        Ox = 25 + Ox
        Ox1 = 25 + Ox1
        hit = 0
        # hit counter
        # coordinates of feature point compare to slice coordinates
        for k in range(0, len(rotated_sova_keypoints_orb)):
            if (Ox <= rotated_sova_keypoints_orb[k].pt[0] <= Ox1) & (Oy <= rotated_sova_keypoints_orb[k].pt[1] <= Oy1):
                hit_orb += 1
                matrix_orb[j][i] = hit_orb
        for s in range(0, len(rotated_sova_keypoints_brisk)):
            if (Ox <= rotated_sova_keypoints_brisk[s].pt[0] <= Ox1) & (Oy <= rotated_sova_keypoints_brisk[s].pt[1] <= Oy1):
                hit_orb += 1
                matrix_brisk[j][i] = hit_orb

for v in range(17):
    for r in range(17):
            matrix_orb_brisk[v][r] = abs(matrix_orb[v][r] & matrix_brisk[v][r])

# Print results
print("orb:")
for n in range(0, len(matrix_orb)):
    print(matrix_orb[n])

print("brisk:")

for n in range(0, len(matrix_brisk)):
    print(matrix_brisk[n])

print("orb and brisk:")

for n in range(0, len(matrix_orb_brisk)):
    print(matrix_orb_brisk[n])

################ 5 DISTANCE ########################################

# create BFMatcher object
distance_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = distance_orb.match(sova_descriptor_orb, rotated_sova_descriptor_orb)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Apply ratio test
#good = []
  #  for m, n in matches:
  #      if m.distance < 0.75*n.distance:
   #      good.append([m])
# Draw first 10 matches.

draw_params = dict(matchColor = (0,255,0),
                   #singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatches(sova_image, sova_keypoints_orb, rotated_sova, rotated_sova_keypoints_orb, matches[:15], None, **draw_params)


plt.imshow(img3), plt.show()
#img3 = cv.drawMatches(sova_image, sova_keypoints_orb, rotated_sova, rotated_sova_keypoints_orb, matches[:30], None, cv.DrawMatchesFlags_DEFAULT)

# Match descriptors.
matches = distance_orb.match(sova_descriptor_orb, img_increased_descriptor_orb)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.

draw_params = dict(matchColor = (0,255,0),
                   #singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatches(sova_image, sova_keypoints_orb, img_increased, img_increased_keypoints_orb, matches[:30], None, **draw_params)
#img3 = cv.drawMatches(sova_image, sova_keypoints_orb, rotated_sova, rotated_sova_keypoints_orb, matches[:30], None, cv.DrawMatchesFlags_DEFAULT)

plt.imshow(img3), plt.show()


matches = distance_orb.match(sova_descriptor_brisk, rotated_sova_descriptor_brisk)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


draw_params = dict(matchColor = (225,36,0),
                   #singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatches(sova_image, sova_keypoints_brisk, rotated_sova, rotated_sova_keypoints_brisk, matches[:15], None, **draw_params)
#img3 = cv.drawMatches(sova_image, sova_keypoints_orb, rotated_sova, rotated_sova_keypoints_orb, matches[:30], None, cv.DrawMatchesFlags_DEFAULT)

plt.imshow(img3), plt.show()

matches = distance_orb.match(sova_descriptor_brisk, img_increased_descriptor_brisk)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


draw_params = dict(matchColor = (225,36,0),
                   #singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatches(sova_image, sova_keypoints_brisk, img_increased, img_increased_keypoints_orb, matches[:20], None, **draw_params)
#img3 = cv.drawMatches(sova_image, sova_keypoints_orb, rotated_sova, rotated_sova_keypoints_orb, matches[:30], None, cv.DrawMatchesFlags_DEFAULT)

plt.imshow(img3), plt.show()