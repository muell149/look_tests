import cv2
import imutils
import dlib

import utils_image

import face_detection.openface_models as openface_models
import face_detection.dlib_models as dlib_models
import face_detection.draw_functions as draw_functions


# calculate_transformations_jitter. 
# Calculate different embeddings using jitter from dlib 
# to create data augmentation
# The np_image has to be one of a face and already aligned
def calculate_embeddings_aligned(
        np_image, 
        embedder,
        num_jitters=5, 
        disturb_colors=True,
        save_transformations=False, 
        embedding_factor=1,
        image_name="img"):

    # Creating augmented images from the sent face
    jittered_images = dlib.jitter_image(np_image, num_jitters, disturb_colors)

    # Storing embedding with original face
    original_embedding = embedder(np_image, embedding_factor)
    embeddings = [ ", ".join([str(i) for i in original_embedding])]

    # Creating embeddings for all jitter transformations
    for i, image in enumerate(jittered_images):
        embedding = embedder(image, embedding_factor)

        # Creating string to be stored
        embedding_string = ", ".join([str(i) for i in embedding])

        if save_transformations:
            # In case the transformations need to be saved on an image
            cv2.imwrite("{}_jitter_{}.jpg".format(image_name, i), image)

        embeddings.append(embedding_string)

    return embeddings


# calculate_transformations
# Creates all the embeddings for all the transformations for a non-aligned image
# The order for the transformations is:
#    - first angle transformations
#    - second blur transformations
#    - third blob transformations
def calculate_embeddings_not_aligned(
        np_image, 
        angle_array=[-20, -10, 0, 10, 20],
        blurs_array=[0, 5],
        blobs_array=[95, 80, 65],
        embedding_factor=1,
        face_confidence=0.8,
        save_transformations=False,
        image_name="img"):

    copy_np_image = np_image.copy()

    # Array to store all the transformations
    embeddings = []

    for blob in blobs_array:
        # Last transformation for the embeddings 
        # The blob size is used to calculate embedding for different image sizes

        for blur in blurs_array:
            # Blur is the second transformation to be applied to the image
            image_blur = (cv2.blur(copy_np_image, (blur, blur), cv2.BORDER_DEFAULT) if blur else copy_np_image)

            for angle in angle_array:
                # Angle is the first transformation to be applied to the image
                image_rotated = (imutils.rotate_bound(image_blur, angle) if angle else image_blur)

                # First the face detection is applied to the transformation
                # If the blur level is to high no face will be detected in
                # image
                detection = openface_models.detect_face_in_image(image_rotated, face_confidence)

                embedding_string = None
                # If there is a face in the image the embeddings are calculated
                if len(detection) > 0:
                    # Transforming the detections to boxes in order to extract the
                    # face from the the image
                    box_detection = openface_models.face_detections_to_box([detection])
                    point1_x, point1_y, point2_x, point2_y = box_detection[0]

                    # Selecting the section of the image where the face was found
                    np_face = image_rotated[point1_y:point2_y, point1_x:point2_x]

                    # Calculating the embedding of the found face
                    embedding = openface_models.calculate_embedding(np_face, factor=embedding_factor, blob_size=blob)

                    # Creating string to be stored
                    embedding_string = ", ".join([str(i) for i in embedding.flatten().tolist()])

                    if save_transformations:
                        # In case the transformations need to be saved as an image
                        image_box = draw_functions.draw_rectangles(image_rotated, box_detection)
                        cv2.imwrite("{}_{}_{}_{}.jpg".format(image_name, blob, blur, angle), image_box)

                embeddings.append(embedding_string)

    return embeddings


# calculate_embeddings_mixed creates all the embeddings for an aligned 
# face. The face is warped and tranformed in order to increase the vector
# space for the selected face
# The order for the transformations is:
#    - first angle transformations
#    - second brightness transformations
#    - third and onwards are the perpective transformations
# Since the transformation is calculated using the aligned face it is
# recommended that the array of tranformations contains small values no 
# larger than 0.05
def calculate_embeddings_mixed(
        np_face, 
        embedder=openface_models.calculate_embedding,
        angle_array=[-2, 0, 2],
        brightness_array=[-20, 0, 20],
        left_delta_array=[0, 0.03],
        right_delta_array=[0, 0.03],
        up_delta_array=[0, 0.03],
        down_delta_array=[0, 0.03],
        embedding_factor=1,
        face_confidence=0.8,
        save_transformations=False,
        image_name="img"):

    # Array to store all the transformations
    embeddings = []

    for angle in angle_array:
        # Angle is the first transformation to be applied to the image
        np_rotated = imutils.rotate(np_face, angle)

        for brigtness in brightness_array:
            # Brightness tranformation on the face
            np_bright = cv2.add(np_rotated, brigtness)

            # Perpective transformation for the face
            for left_delta in left_delta_array:
                for right_delta in right_delta_array:
                    for up_delta in up_delta_array:
                        for down_delta in down_delta_array:

                            # Creating the transformed image from the face
                            np_distorted = utils_image.calculate_perspective(
                                np_bright, left_delta, right_delta, up_delta, down_delta)

                            # Calculating the embedding of the transformed face
                            embedding = embedder(np_distorted, factor=embedding_factor)

                            # Creating string to be stored
                            embedding_string = ", ".join([str(i) for i in embedding.flatten().tolist()])

                            if save_transformations:
                                # In case the transformations need to be saved as an image
                                cv2.imwrite("{}_{}_{}_{}_{}_{}_{}.jpg".format(
                                    image_name, angle, brigtness, left_delta, right_delta, up_delta, down_delta), np_distorted)

                            embeddings.append(embedding_string)

    return embeddings