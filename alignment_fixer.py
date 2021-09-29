import enum
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


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def align_template_to_image_using_affine_transform(template, image):
    return cv2.estimateRigidTransform(template, image, False)


def find_contours(image, threshold, _debug=False):
    # Detect edges using Canny
    canny_output = cv2.Canny(image, threshold, threshold * 2)

    # Find contours
    _, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if _debug:
        filter_and_draw_contours(contours, hierarchy, canny_output.shape, area_threshold=5, _debug=_debug)
    return canny_output


def filter_and_draw_contours(contours, hierarchy, output_shape, area_threshold, _debug=False):
    # Draw contours
    rng.seed(12345)
    drawing = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < area_threshold:
            continue
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color, -1, cv2.LINE_8, hierarchy, 0)
    if _debug:
        cv2.imshow('Contours', drawing)
        cv2.waitKey(0)
    gray_drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    return gray_drawing


def present_overlayed_images(image1, image2):
    print("[DEBUG] presenting aligned images...")
    # resize both the aligned and template images so we can easily
    # visualize them on our screen
    image1_for_visualization = imutils.resize(image1, width=600)
    image2_for_visualization = imutils.resize(image2, width=600)
    # our visualization will be *overlaying* the
    # aligned image on the template, that way we can obtain an idea of
    # how good our image alignment is
    output = image1_for_visualization.copy()
    cv2.addWeighted(image2_for_visualization.copy(), 0.5, output, 0.5, 0, output)
    # show the output image alignment visualization
    cv2.imshow("Image Alignment Overlay", output)
    cv2.waitKey(0)


class PreProcess(enum.Enum):
    NOTHING = 'nothing'
    GAUSSIAN_BLUR = 'GAUSSIAN'
    MEDIAN_BLUR = 'MEDIAN'
    BOX_FILTER_BLUR = 'BOX'
    CONTOUR_STRIP = 'CONTOUR'


def prepare_for_matching(image_to_prepare, kernel, method: PreProcess, _debug=False):
    if method is PreProcess.NOTHING:
        image = image_to_prepare
    elif method is PreProcess.GAUSSIAN_BLUR:
        image = cv2.GaussianBlur(image_to_prepare, (kernel, kernel), 0)
    elif method is PreProcess.MEDIAN_BLUR:
        image = cv2.medianBlur(image_to_prepare, kernel)
    elif method is PreProcess.BOX_FILTER_BLUR:
        image = cv2.blur(image_to_prepare, (kernel, kernel))
    elif method is PreProcess.CONTOUR_STRIP:
        image = cv2.GaussianBlur(image_to_prepare, (kernel, kernel), 0)
    image = unsharp_mask(image)
    image = cv2.threshold(image, 80, 255, cv2.THRESH_TOZERO)[1]
    return find_contours(image, 100, _debug=_debug)


def align_template_to_image(template, image, should_use_homography=False, _debug=False):
    global aligned_template
    preprocessing_method = PreProcess.MEDIAN_BLUR
    kernel_size = 5
    image_for_alignment = prepare_for_matching(image, kernel_size, preprocessing_method, _debug=_debug)
    template_for_alignment = prepare_for_matching(template, kernel_size, preprocessing_method,
        _debug=_debug)
    (h, w) = image.shape[:2]
    if should_use_homography:
        transformation = align_template_to_image_using_homography(template_for_alignment, image_for_alignment,
            _debug=_debug)
        aligned_template = cv2.warpPerspective(template, transformation, (w, h))
        aligned_template_for_presentation = cv2.warpPerspective(template_for_alignment, transformation, (w, h))
    else:
        transformation = align_template_to_image_using_affine_transform(template_for_alignment,
            image_for_alignment)
        if transformation is None:
            print("[ERROR] could not find affine transformation")
            exit(-1)
        aligned_template = cv2.warpAffine(template, transformation, (w, h))
        aligned_template_for_presentation = cv2.warpAffine(template_for_alignment, transformation, (w, h))
    if _debug:
        present_overlayed_images(aligned_template_for_presentation, image_for_alignment)
        present_overlayed_images(aligned_template, image)
    return aligned_template
