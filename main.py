import cv2
import numpy as np
import os
import errno
from os import path
from glob import glob


SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"


def findMatchesBetweenImages(image1, image2, type):
    """
    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (grayscale).
    image_2 : numpy.ndarray
        The second image. (grayscale).

    Returns
    -------
    image_1_kp : list
        The image_1 keypoints, the elements are of type cv2.KeyPoint.
    image_2_kp : list
        The image_2 keypoints, the elements are of type cv2.KeyPoint.
    matches : list
        A list of matches. Each item in the list is of type cv2.DMatch.
    """
    matches = None       # type: list of cv2.DMath
    image_1_kp = None    # type: list of cv2.KeyPoint items
    image_1_desc = None  # type: numpy.ndarray of numpy.uint8 values.
    image_2_kp = None    # type: list of cv2.KeyPoint items.
    image_2_desc = None  # type: numpy.ndarray of numpy.uint8 values.


    if type == "ORB":
        orb = cv2.ORB()
        # orb = cv2.orb = cv2.ORB_create()    # alternate call required on some OpenCV versions
        ## This is for the Investigation part of the assignment only.
        # orb = cv2.ORB(nfeatures=500, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
        #               patchSize=31)
        orb = cv2.ORB()
        image_1_kp, image_1_desc = orb.detectAndCompute(image1, None)
        image_2_kp, image_2_desc = orb.detectAndCompute(image2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(image_1_desc,  image_2_desc)
        matches = sorted(matches, key = lambda x: x.distance)
        matches = matches[:10] # return top 10

    return image_1_kp, image_2_kp, matches


def drawMatches(image_1, image_1_keypoints, image_2, image_2_keypoints, matches):
    """ Draws the matches between the image_1 and image_2.
    Args:
    image_1 (numpy.ndarray): The first image (can be color or grayscale).
    image_1_keypoints (list): The image_1 keypoints, the elements are of type
                              cv2.KeyPoint.
    image_2 (numpy.ndarray): The image to search in (can be color or grayscale)
    image_2_keypoints (list): The image_2 keypoints, the elements are of type
                              cv2.KeyPoint.
    Returns:
    output (numpy.ndarray): An output image that draws lines from the input
                            image to the output image based on where the
                            matching features are.
    """
    # Compute number of channels.
    num_channels = 1
    if len(image_1.shape) == 3:
        num_channels = image_1.shape[2]
    # Separation between images.
    margin = 10
    # Create an array that will fit both images (with a margin of 10 to
    # separate the two images)
    joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                            image_1.shape[1] + image_2.shape[1] + margin,
                            3))
    if num_channels == 1:
        for channel_idx in range(3):
            joined_image[:image_1.shape[0],
                         :image_1.shape[1],
                         channel_idx] = image_1
            joined_image[:image_2.shape[0],
                         image_1.shape[1] + margin:,
                         channel_idx] = image_2
    else:
        joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
        joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

    for match in matches:
        image_1_point = (int(image_1_keypoints[match.queryIdx].pt[0]),
                         int(image_1_keypoints[match.queryIdx].pt[1]))
        image_2_point = (int(image_2_keypoints[match.trainIdx].pt[0] +
                             image_1.shape[1] + margin),
                         int(image_2_keypoints[match.trainIdx].pt[1]))

        rgb = (np.random.rand(3) * 255).astype(np.int)
        cv2.circle(joined_image, image_1_point, 5, rgb, thickness=-1)
        cv2.circle(joined_image, image_2_point, 5, rgb, thickness=-1)
        cv2.line(joined_image, image_1_point, image_2_point, rgb, thickness=3)

    return joined_image



EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])
NAME_KEYS = ['template', 'lighting', 'rotation', 'sample', 'scale', 'other']

subfolders = os.walk(SRC_FOLDER)
subfolders.next()  # skip the root input folder


for dirpath, dirnames, fnames in subfolders:
    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)
    image_files = {}  # map transform name keys to image file names

    print "Processing files in '" + image_dir + "' folder..."

    try:
        for name in NAME_KEYS:
            file_list = reduce(list.__add__, map(glob, [os.path.join(dirpath, '*{}.'.format(name) + ext) for ext in EXTENSIONS]))

            if len(file_list) == 0:
                msg = "  Unable to proceed: no file named {} found in {}"
                raise RuntimeError(msg.format(name, dirpath))
            elif len(file_list) > 1:
                msg = "  Unable to proceed: too many files matched the pattern `*{}.EXT` in {}"
                raise RuntimeError(msg.format(name, dirpath))

            image_files[name] = file_list[0]

    except RuntimeError as err:
        print err
        continue  # skip this folder

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


template = cv2.imread(image_files['template'])
del image_files['template']

for transform_name in image_files:
    print "    Processing '{}' image".format(transform_name)
    image = cv2.imread(image_files[transform_name])
    keypoints1, keypoints2, matches = findMatchesBetweenImages(template, image, "ORB")
    annotated_matches = drawMatches(template, keypoints1, image, keypoints2, matches)
    cv2.imwrite(path.join(output_dir, transform_name + '.jpg'), annotated_matches)

