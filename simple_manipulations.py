def crop_image(image, x_offset, y_offset, sliding_window_size):
    return \
        image[
        x_offset:min(x_offset + sliding_window_size, image.shape[1]),
        y_offset:min(y_offset + sliding_window_size, image.shape[0])
        ]