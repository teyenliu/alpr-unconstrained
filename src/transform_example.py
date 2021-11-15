# import the necessary packages
from transform import four_point_transform
from transform import four_point_transform_and_replacement
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-r", "--replace", help = "path to the image file for replacement")
ap.add_argument("-c", "--coords",
    help = "comma seperated list of source points")
args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype = "float32")
# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)
# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)

rep_image = cv2.imread(args["replace"])
reverse_warped = four_point_transform_and_replacement(image, pts, rep_image)

#result_image = cv2.bitwise_and(reverse_warped, image)
cv2.imshow("Reverse_Warped", reverse_warped)
cv2.imwrite('./output.jpg', reverse_warped)

cv2.waitKey(0)
