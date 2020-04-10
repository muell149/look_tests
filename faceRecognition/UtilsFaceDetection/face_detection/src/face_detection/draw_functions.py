import base64
import numpy as np
import cv2

# Modules with the face detection models from two libraries
import face_detection.dlib_models as dlib_models

def draw_points_faces(np_image, size=2):
    # The dlib framework uses a detection object as a dlib.rectangle
    # This rectangle for this function is the whole image since 
    # the accepted image is supposed to be only of a face
    detections = dlib_models.face_detector(np_image, 1)

    if len(detections) == 0:
        return None

    np_image_copy = np_image.copy()

    for detection in detections:
        # Calculate the 5 landmarks in a face
        shape = dlib_models.face_shape(np_image, detection)

        for i, point in enumerate(shape.parts()):
            point_x, point_y = point.x, point.y

            cv2.circle(
                np_image_copy, 
                (point_x, point_y), 
                size, 
                (255,0,0), 
                -1)

    return np_image_copy

# draw_box_faces
# Utility function to draw rectangles based on the detections found in 
# the image
# The rectangles to be drawn must follow this arrangement
# [[box1_x1, box1_y1, box1_x2, box1_y2],
#  [box2_x1, box2_y1, box2_x2, box2_y2],
#  [box3_x1, box3_y1, box3_x2, box3_y2]]
def draw_rectangles(np_image, rectangles, color=(0, 0, 255), line_width=1):

    copy_np_image = np_image.copy()

    # In case a flat array is sent to the draw function
    if len(rectangles) == 0:
        return copy_np_image

    for box in rectangles:

        # If there is an empty detection skip the drawing process
        if len(box) == 0:
            break

        # Calculating the area where the face is
        (point1_x, point1_y, point2_x, point2_y) = [int(val) for val in box]

        cv2.rectangle(
            copy_np_image,
            (point1_x, point1_y),
            (point2_x, point2_y),
            color,
            line_width
        )

    return copy_np_image


# create_thumbnail. Creates a thumbnail from a numpy image.
# The size of the thumbnail is controlled by the parameter
# IMAGE_THUMBNAIL_DELTA[X,Y] from the config.ini file
def create_thumbnail(
        np_image,
        box,
        delta_x=0.1,
        delta_y=0.1,
        thumb_size=80):

    # Selecting the points where the face is located
    point1_x, point1_y, point2_x, point2_y = [int(val) for val in box]

    delta_x = int((point2_x - point1_x) * delta_x)
    delta_y = int((point2_y - point1_y) * delta_y)

    # Thumbnail area selection based on the calculated deltas
    thumbnail = np_image[
        max(0, point1_y - delta_y) : max(0, point2_y + delta_y),
        max(0, point1_x - delta_x) : max(0, point2_x + delta_x),
        :]

    # Resizing image
    thumbnail = cv2.resize(
        thumbnail,
        dsize = (thumb_size, thumb_size),
        interpolation = cv2.INTER_CUBIC)

    _, buffer = cv2.imencode(".jpg", thumbnail)
    thumbnail_string = base64.b64encode(buffer).decode("utf-8")

    # Write to a file to show conversion worked
    # Example to return the encoded string back to a jpg file
    #jpg_original = base64.b64decode(bytes(thumbnail_string, "utf"))
    #with open("test.jpg", "wb") as f_output:
        #f_output.write(jpg_original)

    return thumbnail_string
