import argparse
import math
import random as rng

import cv2
import imutils
import numpy as np


def euclidian_distance(pt1: tuple, pt2: tuple):
    return math.sqrt(sum([(c1 - c2) ** 2 for c1, c2 in zip(pt1, pt2)]))


# this function does not makes sense yet. this threshold needs to be calibrated somehow if I am ever to use it.
def is_match_sensible(match, image_feature_keys, template_feature_keys):
    return euclidian_distance(template_feature_keys[match.queryIdx].pt, image_feature_keys[match.trainIdx].pt) < 30


def align_template_to_image_using_homography(template, image, maxFeatures=500, keepPercent=0.2, _debug=False):
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    patch_size = 5
    orb = cv2.ORB_create(maxFeatures, patchSize=patch_size + 2, edgeThreshold=patch_size)
    (image_feature_keys, image_feature_descriptors) = orb.detectAndCompute(image, None)
    (template_feature_keys, template_feature_descriptors) = orb.detectAndCompute(template, None)

    # match the features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(template_feature_descriptors, image_feature_descriptors, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    # keep = int(len(matches) * keepPercent)
    # matches = matches[:keep]
    matches = [match for match in matches if is_match_sensible(match, image_feature_keys, template_feature_keys)]
    # check to see if we should visualize the matched keypoints
    if _debug:
        visual_matches = cv2.drawMatches(template, template_feature_keys, image, image_feature_keys,
            matches, None)
        visual_matches = imutils.resize(visual_matches, width=1000)
        cv2.imshow("Matched Keypoints", visual_matches)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    match_coordinates_in_image = np.zeros((len(matches), 2), dtype="float")
    match_coordinates_in_template = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (index, matches) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        match_coordinates_in_template[index] = template_feature_keys[matches.queryIdx].pt
        match_coordinates_in_image[index] = image_feature_keys[matches.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(match_coordinates_in_template, match_coordinates_in_image, method=cv2.RANSAC)
    return H


def align_template_to_image_using_affine_transform(template, image):
    return cv2.estimateRigidTransform(template, image, True)


def find_contours(image, threshold, _debug=False):
    rng.seed(12345)
    # Detect edges using Canny
    # image = cv2.medianBlur(image, 13)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    canny_output = cv2.Canny(image, threshold, threshold * 2)
    # Find contours
    _, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    if _debug:
        cv2.imshow('Contours', drawing)
        cv2.waitKey(0)
    gray_drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    return gray_drawing


if __name__ == "__main__":
    show_debug_windows = True
    use_homography = False
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image that we'll align to template")
    ap.add_argument("-t", "--template", required=True,
        help="path to input template image")
    args = vars(ap.parse_args())

    # load the input image and template from disk
    print("[INFO] loading images...")
    image = cv2.imread(args["image"])
    template = cv2.imread(args["template"])
    # align the images
    print("[INFO] aligning images...")

    # convert both the input image and template to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    image_contours = find_contours(image, 100, _debug=show_debug_windows)
    template_contours = find_contours(template, 100, _debug=show_debug_windows)

    (h, w) = image.shape[:2]
    if use_homography:
        transformation = align_template_to_image_using_homography(template_contours, image_contours,
            _debug=show_debug_windows)
        aligned_template = cv2.warpPerspective(template, transformation, (w, h))
    else:
        transformation = align_template_to_image_using_affine_transform(template_contours, image_contours)
        aligned_template = cv2.warpAffine(template, transformation, (w, h))

    if show_debug_windows:
        print("[DEBUG] presenting aligned images...")
        # resize both the aligned and template images so we can easily
        # visualize them on our screen
        aligned_template_for_visualization = imutils.resize(aligned_template, width=700)
        image_for_visualization = imutils.resize(image, width=700)

        # our visualization will be *overlaying* the
        # aligned image on the template, that way we can obtain an idea of
        # how good our image alignment is
        overlay = image_for_visualization.copy()
        output = aligned_template_for_visualization.copy()
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
        # show the output image alignment visualization
        cv2.imshow("Image Alignment Overlay", output)
        cv2.waitKey(0)

    print("[INFO] diffing images...")
    area_of_interest = cv2.threshold(aligned_template, 1, 255, cv2.THRESH_BINARY)[1]
    image_cropped_for_interest = np.where(area_of_interest, image, area_of_interest)

    # compute the absolute difference between the current frame and
    # first frame
    image_delta = cv2.absdiff(image, aligned_template)
    image_delta = np.where(area_of_interest, image_delta, area_of_interest)
    possible_defects = cv2.threshold(image_delta, 25, 255, cv2.THRESH_BINARY)[1]
    possible_defects = cv2.bitwise_and(possible_defects, area_of_interest)

    if show_debug_windows:
        cv2.imshow("Image Difference", image_delta)
        cv2.imshow("Difference Mask", possible_defects)
        cv2.waitKey(0)
