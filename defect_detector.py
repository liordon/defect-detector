import argparse

import cv2
import numpy as np

from alignment_fixer import find_matching_point_between_patch_and_reference, filter_and_draw_contours, \
    align_template_to_image
from simple_manipulations import crop_image_by_size, crop_image_by_coordinates


def could_be_defect(contour, defect_size_threshold=20):
    (x, y, w, h) = cv2.boundingRect(contour)
    return cv2.contourArea(contour) > defect_size_threshold and not ((w / h > 2) or (h / w > 2))


def find_defects_by_diffing_images(image, aligned_template, _debug=False):
    print("[INFO] diffing images...")
    area_of_interest = cv2.threshold(aligned_template, 0, 255, cv2.THRESH_BINARY)[1]
    area_of_interest = cv2.erode(area_of_interest, np.ones((21, 21), np.uint8), iterations=3)
    # compute the absolute difference between the images
    image_delta = cv2.absdiff(image, aligned_template)
    image_delta = np.where(area_of_interest, image_delta, area_of_interest)
    possible_defects = cv2.threshold(image_delta, 35, 255, cv2.THRESH_BINARY)[1]
    possible_defects = cv2.bitwise_and(possible_defects, area_of_interest)
    if _debug:
        cv2.imshow("Image Difference", image_delta)
        cv2.imshow("Difference Mask", possible_defects)
        cv2.waitKey(0)
    return possible_defects


def search_for_defects_in_sliding_windows(image_path, template_path, sliding_window_size=200, sliding_window_steps=50,
        _debug=False):
    # load the input image and template from disk
    print("[INFO] loading images...")
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    if image is None or template is None:
        print("[ERROR] image to reference could not be loaded. please check paths.")
        exit(-1)

    # align the images
    print("[INFO] aligning images...")
    # convert both the input image and template to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    image = crop_image_by_reference_coverage(image, template)

    if _debug:
        cv2.imshow("image cropped according to coverage", image)


    accumulated_defects = np.ones(image.shape)

    # for height_offset in range(0, image.shape[0] - int(sliding_window_size / 2), sliding_window_steps):
    #     for width_offset in range(0, image.shape[1] - int(sliding_window_size / 2), sliding_window_steps):
    #         # extra_width = (1 if x_offset % sliding_window_size == 0 and width_offset % sliding_window_size == 0 else 0)
    #         # cv2.rectangle(image,
    #         #     (x_offset, width_offset),
    #         #     (x_offset + sliding_window_size, width_offset + sliding_window_size),
    #         #     (150,0,0), 1 + extra_width)
    #         # continue
    #         print(f"[INFO] inspecting {height_offset}, {width_offset} window")
    #         cropped_image = crop_image_by_size(image, height_offset, width_offset, sliding_window_size)

            # minLoc, _ = find_matching_point_between_patch_and_reference(template, cropped_image, _debug=_debug)
            # (h, w) = cropped_image.shape[:2]
            # cropped_template = template[minLoc[1]:minLoc[1] + h, minLoc[0]:minLoc[0] + w]
            # if cropped_template is None:
            #     continue
    cropped_image = image
    minLoc, _ = find_matching_point_between_patch_and_reference(template, cropped_image, _debug=_debug)
    (h, w) = cropped_image.shape[:2]
    cropped_template = template[minLoc[1]:minLoc[1] + h, minLoc[0]:minLoc[0] + w]

    possible_defects = find_defects_by_diffing_images(cropped_template, image,
        _debug=_debug)

    _, contours, hierarchy = cv2.findContours(possible_defects, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    defect_contours = [contour for contour in contours if could_be_defect(contour,defect_size_threshold=3)]
    gray_drawing = filter_and_draw_contours(defect_contours, hierarchy, cropped_image.shape, area_threshold=0,
        _debug=_debug)

    # for i in range(cropped_image.shape[1]):
    #     for j in range(cropped_image.shape[1]):
    #         accumulated_defects[height_offset + i, width_offset+j] *= gray_drawing[i,j]
    accumulated_defects = gray_drawing

    cv2.imshow("Result", accumulated_defects)
    cv2.waitKey(0)

    boolean_image = cv2.threshold(accumulated_defects, 1, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("windows", boolean_image)
    cv2.waitKey(0)


def crop_image_by_reference_coverage(image, reference, _debug=False):
    (h, w) = image.shape[:2]

    quarter_image_size = int(min(image.shape[:2]) / 4)
    image_left = crop_image_by_size(image, quarter_image_size, 0, quarter_image_size * 2)
    template_left = crop_image_by_size(reference, quarter_image_size, 0, quarter_image_size * 2)
    minImageLoc, minImageVal = find_matching_point_between_patch_and_reference(reference, image_left, _debug=_debug)
    minTemplateLoc, minTemplateVal = find_matching_point_between_patch_and_reference(image, template_left,
        _debug=_debug)
    left_image_edge = 0 if minImageVal < minTemplateVal else minTemplateLoc[0]

    image_top = crop_image_by_size(image, 0, quarter_image_size, quarter_image_size * 2)
    template_top = crop_image_by_size(reference, 0, quarter_image_size, quarter_image_size * 2)
    minImageLoc, minImageVal = find_matching_point_between_patch_and_reference(reference, image_top, _debug=_debug)
    minTemplateLoc, minTemplateVal = find_matching_point_between_patch_and_reference(image, template_top, _debug=_debug)
    top_image_edge = 0 if minImageVal < minTemplateVal else minTemplateLoc[1]

    image_right = crop_image_by_size(image, quarter_image_size, w - quarter_image_size * 2, quarter_image_size * 2)
    template_right = crop_image_by_size(reference, quarter_image_size, w - quarter_image_size * 2, quarter_image_size * 2)
    minImageLoc, minImageVal = find_matching_point_between_patch_and_reference(reference, image_right, _debug=_debug)
    minTemplateLoc, minTemplateVal = find_matching_point_between_patch_and_reference(image, template_right,
        _debug=_debug)
    right_image_edge = w if minImageVal < minTemplateVal else minTemplateLoc[0] + quarter_image_size * 2

    image_bottom = crop_image_by_size(image, h - quarter_image_size * 2, quarter_image_size, quarter_image_size * 2)
    template_bottom = crop_image_by_size(reference, h - quarter_image_size * 2, quarter_image_size, quarter_image_size * 2)
    minImageLoc, minImageVal = find_matching_point_between_patch_and_reference(reference, image_bottom, _debug=_debug)
    minTemplateLoc, minTemplateVal = find_matching_point_between_patch_and_reference(image, template_bottom,
        _debug=_debug)
    bottom_image_edge = h if minImageVal < minTemplateVal else minTemplateLoc[1] + quarter_image_size * 2

    if _debug:
        print(f"Original image dims: {(h, w)}")
        print(f"Cropped image limits: {(top_image_edge, left_image_edge), (bottom_image_edge, right_image_edge)}")

        image_with_rectangle = image.copy()
        cv2.rectangle(
            image_with_rectangle,
            (left_image_edge, top_image_edge),
            (right_image_edge, bottom_image_edge),
            (255,255,255),
            thickness=2
        )
        cv2.imshow("Image marked for coverage crop", image_with_rectangle)
        cv2.waitKey(0)

    return crop_image_by_coordinates(image, top_image_edge, bottom_image_edge, left_image_edge, right_image_edge)


if __name__ == "__main__":
    show_debug_windows = True
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image that we'll align to template")
    ap.add_argument("-t", "--template", required=True,
        help="path to input template image")
    ap.add_argument("-d", "--debug", required=False, default=show_debug_windows,
        help="display partial results and detection process snapshots")
    args = vars(ap.parse_args())

    search_for_defects_in_sliding_windows(args["image"], args["template"], _debug=args["debug"])
