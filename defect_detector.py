import argparse

import cv2
import numpy as np

from alignment_fixer import align_template_to_image, filter_and_draw_contours


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

    for x_offset in range(0, image.shape[0] - sliding_window_size, sliding_window_steps):
        for y_offset in range(0, image.shape[1] - sliding_window_size, sliding_window_steps):
            print(f"[INFO] inspecting {x_offset}, {y_offset} window")
            cropped_image = image[x_offset:x_offset + sliding_window_size, y_offset:y_offset + sliding_window_size]
            cropped_template = template[x_offset:x_offset + sliding_window_size,
            y_offset:y_offset + sliding_window_size]

            aligned_template = align_template_to_image(cropped_template, cropped_image, _debug=_debug)
            if aligned_template is None:
                continue

            possible_defects = find_defects_by_diffing_images(cropped_image, aligned_template,
                _debug=_debug)

            _, contours, hierarchy = cv2.findContours(possible_defects, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            defect_contours = [contour for contour in contours if could_be_defect(contour)]
            gray_drawing = filter_and_draw_contours(defect_contours, hierarchy, image.shape, area_threshold=0,
                _debug=_debug)
            boolean_image = cv2.threshold(gray_drawing, 1, 255, cv2.THRESH_BINARY)[1]

            cv2.imshow("Result", boolean_image)
            cv2.waitKey(0)


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
