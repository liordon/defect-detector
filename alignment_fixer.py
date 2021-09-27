import argparse

import cv2
import imutils
import numpy as np


def align_template_to_image(template, image, maxFeatures=500, keepPercent=0.2, _debug=False):
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (image_feature_keys, image_feature_descriptors) = orb.detectAndCompute(image, None)
    (template_feature_keys, template_feature_descriptors) = orb.detectAndCompute(template, None)

    # match the features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(image_feature_descriptors, template_feature_descriptors, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if _debug:
        visual_matches = cv2.drawMatches(image, image_feature_keys, template, template_feature_keys,
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
        match_coordinates_in_image[index] = image_feature_keys[matches.queryIdx].pt
        match_coordinates_in_template[index] = template_feature_keys[matches.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(match_coordinates_in_template, match_coordinates_in_image, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = image.shape[:2]
    aligned_template = cv2.warpPerspective(template, H, (w, h))
    # return the aligned image
    return aligned_template


if __name__ == "__main__":
    show_debug_reports = True
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

    aligned_template = align_template_to_image(template, image, _debug=show_debug_reports)

    if show_debug_reports:
        # resize both the aligned and template images so we can easily
        # visualize them on our screen
        aligned_template_for_visualization = imutils.resize(aligned_template, width=700)
        image_for_visualization = imutils.resize(image, width=700)

        # our first output visualization of the image alignment will be a
        # side-by-side comparison of the output aligned image and the
        # template
        stacked_for_visualization = np.hstack([aligned_template_for_visualization, image_for_visualization])

        # our second image alignment visualization will be *overlaying* the
        # aligned image on the template, that way we can obtain an idea of
        # how good our image alignment is
        overlay = image_for_visualization.copy()
        output = aligned_template_for_visualization.copy()
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
        # show the two output image alignment visualizations
        cv2.imshow("Image Alignment Stacked", stacked_for_visualization)
        cv2.imshow("Image Alignment Overlay", output)
        cv2.waitKey(0)

    area_of_interest = cv2.threshold(aligned_template, 1, 255, cv2.THRESH_BINARY)[1]
    image_cropped_for_interest = np.where(area_of_interest, image, area_of_interest)

    # compute the absolute difference between the current frame and
    # first frame
    image_delta = cv2.absdiff(image, aligned_template)
    image_delta = np.where(area_of_interest, image_delta, area_of_interest)
    possible_defects = cv2.threshold(image_delta, 25, 255, cv2.THRESH_BINARY)[1]
    possible_defects = cv2.bitwise_and(possible_defects, area_of_interest)

    if show_debug_reports:
        cv2.imshow("Image Difference", image_delta)
        cv2.imshow("Difference Mask", possible_defects)
        cv2.waitKey(0)
