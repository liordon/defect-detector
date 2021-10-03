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


def align_template_to_image(template, image, _debug=False):
    global aligned_template
    preprocessing_method = PreProcess.MEDIAN_BLUR
    kernel_size = 5
    image_for_alignment = prepare_for_matching(image, kernel_size, preprocessing_method, _debug=_debug)
    template_for_alignment = prepare_for_matching(template, kernel_size, preprocessing_method,
        _debug=_debug)
    (h, w) = image.shape[:2]
    transformation = cv2.estimateRigidTransform(template_for_alignment, image_for_alignment, False)
    if transformation is None:
        print("[ERROR] could not find affine transformation")
        return None
    aligned_template = cv2.warpAffine(template, transformation, (w, h))
    aligned_template_for_presentation = cv2.warpAffine(template_for_alignment, transformation, (w, h))
    if _debug:
        present_overlayed_images(aligned_template_for_presentation, image_for_alignment)
        present_overlayed_images(aligned_template, image)
    return aligned_template


def find_matching_point_between_patch_and_reference(reference, image_patch, _debug=False):
    preprocessing_method = PreProcess.MEDIAN_BLUR
    kernel_size = 5

    if _debug:
        cv2.imshow("template for alignment", reference)
        cv2.imshow("image for alignment", image_patch)
        cv2.waitKey(0)
    patch_for_alignment = image_patch#prepare_for_matching(image_patch, kernel_size, preprocessing_method, _debug=_debug)
    reference_for_alignment = reference#prepare_for_matching(reference, kernel_size, preprocessing_method,
        # _debug=_debug)
    (h, w) = image_patch.shape[:2]
    correlations = cv2.matchTemplate(reference_for_alignment, patch_for_alignment, cv2.TM_SQDIFF)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(correlations)
    aligned_image = np.zeros(reference.shape, dtype=np.uint8)
    cropped_template = reference[minLoc[1]:minLoc[1] + h, minLoc[0]:minLoc[0] + w]
    for i in range(h):
        for j in range(w):
            aligned_image[i + minLoc[1], j + minLoc[0]] = image_patch[i, j]
    if _debug:
        alignment_mask = np.zeros(reference.shape, dtype=np.uint8)
        for i in range(correlations.shape[0]):
            for j in range(correlations.shape[1]):
                alignment_mask[i, j] = int(correlations[i, j] * 255 / maxVal)
        present_overlayed_images(alignment_mask, reference)
        present_overlayed_images(reference, aligned_image)
        cv2.imshow("cropped_template", cropped_template)
        cv2.waitKey(0)
    return minLoc, minVal
