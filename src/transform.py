# import the necessary packages
import numpy as np
import cv2
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def four_point_transform_and_replacement(image, pts, rep_image):
    img_h, img_w = image.shape[:2]
    warped = four_point_transform(image, pts)
    warped_h, warped_w = warped.shape[:2]
    
    # width * height * channel for cv2.resize
    new_rep_image = cv2.resize(rep_image, (warped_w, warped_h), 
            interpolation=cv2.INTER_LINEAR)
    
    src_rect = np.array([
        [0, 0],
        [warped_w - 1, 0],
        [warped_w - 1, warped_h - 1],
        [0, warped_h - 1]], dtype = "float32")

    # obtain a consistent order of the points and unpack them
    # individually
    dst_rect = order_points(pts)
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    reverse_warped = cv2.warpPerspective(new_rep_image, M, (img_w, img_h))
    
    # loop over the image, pixel by pixel
    T = 0 #threshold
    #ret_img = image

    if len(image.shape) == 3:
        #channel = 3, colorful image
        for y in range(0, img_h):
            for x in range(0, img_w):
                # threshold the pixel
                if reverse_warped[y, x][0] > T:
                    image[y, x][0] = reverse_warped[y, x][0]
                    image[y, x][1] = reverse_warped[y, x][1]
                    image[y, x][2] = reverse_warped[y, x][2]
    else:
        #channel = 1, grayscale image
        gray_reverse_warped = cv2.cvtColor(reverse_warped, cv2.COLOR_BGR2GRAY)
        for y in range(0, img_h):
            for x in range(0, img_w):
                # threshold the pixel
                if gray_reverse_warped[y, x] > T:
                    image[y, x] = gray_reverse_warped[y, x]
    return image
