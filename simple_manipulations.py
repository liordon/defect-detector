def crop_image_by_size(image, height_offset, width_offset, sliding_window_size):
    return \
        image[
        height_offset:min(height_offset + sliding_window_size, image.shape[0]),
        width_offset:min(width_offset + sliding_window_size, image.shape[1])
        ]


def crop_image_by_coordinates(image, top_edge, bottom_edge, left_edge, right_edge):
    return \
        image[
        top_edge:bottom_edge,
        left_edge:right_edge
        ]
