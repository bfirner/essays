# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.

import argparse

import numpy
import cv2

from sklearn import svm

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv2.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"
cv_version = cv2.__version__.split('.')[:2]
if cv_version[0] != "4" or int(cv_version[1]) < 10:
    print("This requires openCV version >= 4.10")
    quit(0)



def visualize(image, bbox, score, isLocated, fps=None, box_color=(0, 255, 0),text_color=(0, 255, 0), fontScale = 1, fontSize = 1):
    # TODO FIXME Delete
    output = image.copy()
    h, w, _ = output.shape

    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 30), cv2.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)

    if isLocated and score >= 0.3:
        # bbox: Tuple of length 4
        x, y, w, h = bbox
        cv2.rectangle(output, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(output, '{:.2f}'.format(score), (x, y+25), cv2.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)
    else:
        text_size, baseline = cv2.getTextSize('Target lost!', cv2.FONT_HERSHEY_DUPLEX, fontScale, fontSize)
        text_x = int((w - text_size[0]) / 2)
        text_y = int((h - text_size[1]) / 2)
        cv2.putText(output, 'Target lost!', (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, fontScale, (0, 0, 255), fontSize)

    return output

class Vectorizer:
    def __init__(self, model):
        self._model = cv2.dnn.readNet(model)
        self._input_names = ''
        # These are the features rather than the classes (which are in scale_0.tmp_0)
        self._output_names = ['save_infer_model/scale_1.tmp_0']
        self._expected_dim = 224
        # Images must be preprocessed in the same manner that they were trained
        # These constants are used for normalization. The means are suspect as
        # we are using a different camera, but we'll just hope it works.
        self._mean = numpy.array([[[0.485, 0.456, 0.406]]])
        self._std = numpy.array([[[0.229, 0.224, 0.225]]])

    def inDim(self):
        return self._expected_dim

    def getFeatures(self, image):
        # TODO FIXME Work on a batch of inputs
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # TODO FIXME This is the default processing, but let's allow the user to do some of this
        w, h, _ = image.shape
        if w != self._expected_dim or h != self._expected_dim:
            image = cv2.resize(image, dsize=(self._expected_dim, self._expected_dim))

        # Inputs should be floats in the range [0,1]
        image = (image.astype(numpy.float32, copy=False) / 255.0 - self._mean) / self._std
        input_blob = cv2.dnn.blobFromImage(image)
        self._model.setInput(input_blob, self._input_names)
        features = self._model.forward(self._output_names)
        return features


def run_labeller():
    parser = argparse.ArgumentParser(
        description="VIT dataset labeller")
    parser.add_argument(
        '--vit_model',
        type=str,
        default='object_tracking_vittrack_2023sep.onnx',
        help='Path to the vit model')
    parser.add_argument(
        '--class_model',
        type=str,
        default='image_classification_ppresnet50_2022jan.onnx',
        help='Path to the classifying model.')
    args = parser.parse_args()

    # Initialize the tracking model
    params = cv2.TrackerVit_Params()
    params.net = args.vit_model
    tracker = cv2.TrackerVit_create(params)

    # Initialize the classifier.
    # It has fewer convenience wrappers, so we made our own.
    vectorizer = Vectorizer(args.class_model)

    # Some display constants
    text_yellow=(0, 255, 255)
    text_blue=(255, 0, 0)
    text_green=(0, 255, 0)
    text_red=(0, 0, 255)


    # Open the webcam
    video = cv2.VideoCapture(0)

    # Remember what we are doing.
    # waiting: the user needs to draw a bounding box and start training
    # tracking: the object is being tracked
    # classifying: classifying with the SVM, waiting to draw a new bounding box
    # quitting: the user wants to exit
    mode = "waiting"

    # OpenCV's waitKey function returns not just the key, but also any keyboard modifiers. This
    # means that the returned value cannot be compared to just the key.
    def isKey(cvkey, key):
        return key == (cvkey & 0xFF)

    # Keep going until capture failes or the user quits
    has_frame, frame = video.read()

    positive_examples = []
    negative_examples = []

    if not has_frame:
        print("Capture failures, exiting!")
        exit(1)

    while mode != "quitting" and has_frame:
        display_frame = frame.copy()
        frame_h, frame_w, _ = display_frame.shape
        if mode == "waiting":
            cv2.putText(display_frame, "Press 's' to bbox a target.", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)
        elif mode == "tracking":
            found, foundbbox = tracker.update(frame)
            score = tracker.getTrackingScore()

            if found:
                x, y, w, h = foundbbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), text_yellow, 2)
                cv2.putText(display_frame, f"Score: {score}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)
                cv2.putText(display_frame, "Press any key to stop tracking.", (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)

                # Add the found area to the positive examples and select something else for a negative example
                # We don't need to store the images, only the feature vectors
                # Sizes must be 224 by 224 or they will be resized
                # TODO FIXME Make 224x224 boxes with a positive and negative example
                left = x
                top = y
                if w < vectorizer.inDim():
                    # Widen the bbox
                    extra = vectorizer.inDim() - w
                    left = x - extra//2
                    if left < 0:
                        left = 0
                if h < vectorizer.inDim():
                    # Heighten the bbox
                    extra = vectorizer.inDim() - h
                    top = top - extra//2
                    if top < 0:
                        top = 0
                # TODO FIXME If they are wider than the inDim, use cv2.resize
                positive_image = frame[top:top+vectorizer.inDim(), left:left+vectorizer.inDim(), :]

                # Make a negative example from a different part of the image
                # Check the margins left over from the *tracking bounding box* at the left, right, top, and bottom to see where we can put it
                lmargin = x
                rmargin = frame_w - (x + w)
                if lmargin < rmargin:
                    # Begin to the right of the bounding box
                    nleft = x + w
                else:
                    # Begin to the left of the bounding box
                    nleft = x - vectorizer.inDim()
                tmargin = y
                bmargin = frame_h - (y + h)
                if tmargin < bmargin:
                    # Begin below the bounding box
                    ntop = y + h
                else:
                    # Begin above the bounding box
                    ntop = y - vectorizer.inDim()
                print(f"bbox was {foundbbox}")
                print(f"left and top are {left} and {top} and image is size {frame_w} by {frame_h}")
                print(f"nleft and ntop are {nleft} and {ntop} and image is size {frame_w} by {frame_h}")
                negative_image = frame[ntop:ntop+vectorizer.inDim(), nleft:nleft+vectorizer.inDim(), :]

                positive_examples.append(vectorizer.getFeatures(positive_image))
                negative_examples.append(vectorizer.getFeatures(negative_image))
            else:
                if 0 < len(positive_examples):
                    # Stop tracking and train the svm, then switch to classifying mode.
                    labels = [1] * len(positive_examples) + [0] * len(negative_examples)
                    clf = svm.SVC()
                    clf.fit(positive_examples + negative_examples, labels)
                    mode = "classifying"
                else:
                    mode = "waiting"
        elif mode == "classifying":
            # TODO Classify
            # TODO Begin with a center crop, but move on to a tiled tracking window 
            center_left = frame_h - vectorizer.inDim()//2
            center_top = frame_w - vectorizer.inDim()//2
            center_right = center_left + vectorizer.inDim()
            center_bottom = center_top + vectorizer.inDim()
            center_window = frame[center_top:center_bottom, center_left:center_right]
            features = vectorizer.getFeatures(center_window)
            prediction = clf.predict(features)

            cv2.rectangle(display_frame, (center_left, center_top), (center_right, center_bottom), text_yellow, 2)
            cv2.putText(display_frame, f"Prediction: {prediction}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)
            cv2.putText(display_frame, "Press 's' to bbox a target.", (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)


        cv2.imshow('Label Demo', display_frame)

        key = cv2.waitKey(1)
        if key > 0:
            if isKey(key, ord('q')):
                mode = "quitting"
            elif isKey(key, ord('s')):
                if mode != "tracking":
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "Select the object of interest to begin tracking", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)
                    cv2.imshow('Label Demo', display_frame)
                    roi = cv2.selectROI('Label Demo', display_frame)
                    # Initialize the tracker using the original frame (i.e. without the text on it)
                    tracker.init(frame, roi)
                    mode = "tracking"
            elif mode == "tracking":
                # Stop tracking and train the svm, then switch to classifying mode.
                labels = [1] * len(positive_examples) + [0] * len(negative_examples)
                clf = svm.SVC()
                clf.fit(positive_examples + negative_examples, labels)
                mode = "classifying"

        # Grab the next frame
        has_frame, frame = video.read()

    # Cleanup
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_labeller()
