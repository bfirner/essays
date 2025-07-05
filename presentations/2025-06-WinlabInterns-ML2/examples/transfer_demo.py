#!/usr/bin/python3

# This is an example demonstrating the reuse of a model for transfer learning.

import argparse
import cv2
import numpy
import random
import torch

from class_names import LABELS_IMAGENET_1K

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv2.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"
cv_version = cv2.__version__.split('.')[:2]
if cv_version[0] != "4" or int(cv_version[1]) < 10:
    print("This requires openCV version >= 4.10")
    quit(0)


class PPResnet:
    """This class encapsulates model inference to extract a feature vector."""
    def __init__(self, model):
        self._model = cv2.dnn.readNet(model)
        self._input_names = ''
        # These are the 1000 classes
        self._output_names = ['save_infer_model/scale_0.tmp_0']
        self._expected_dim = 224
        # Images must be preprocessed in the same manner that they were trained
        # These constants are used for normalization. The means are suspect as
        # we are using a different camera, but we'll just hope it works.
        self._mean = numpy.array([[[0.485, 0.456, 0.406]]])
        self._std = numpy.array([[[0.229, 0.224, 0.225]]])

    def inDim(self):
        return self._expected_dim

    def getClassScores(self, image):
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
        class_values = self._model.forward(self._output_names)
        return class_values

def tiny_preprocess(image):
    # Resize to the tiny network's input and use PPResnet's normalization
    mean = numpy.array([[[0.485, 0.456, 0.406]]])
    std = numpy.array([[[0.229, 0.224, 0.225]]])
    image = cv2.resize(image, dsize=(56, 56))
    image = (image.astype(numpy.float32, copy=False) / 255.0 - mean) / std
    # Channels first
    image = numpy.moveaxis(image, -1, 0)
    return image

imagenet_labels = LABELS_IMAGENET_1K.splitlines()
transfer_labels = [
    "tabby",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "robin",
    "jay",
    "chickadee",
    "coffee mug",
    "fountain pen",
    "ballpoint",
    "jean",
    "sunscreen",
    "sweatshirt",
    "monitor",
    "tripod",
    "cellular telephone",
    "water bottle",
    "water jug",
    "wall clock",
    "wallet",
    "analog clock",
    "pole",
    "convertible",
    "passenger car",
    ]


# Input is a 56x56 image
#tiny = torch.nn.Sequential(
#    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=0, stride=1),  # 16x52x52
#    torch.nn.BatchNorm2d(16),
#    torch.nn.ReLU(),
#    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0, stride=1),  # 32x48x48
#    torch.nn.BatchNorm2d(32),
#    torch.nn.ReLU(),
#    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0, stride=1), # 32x44x44
#    torch.nn.BatchNorm2d(32),
#    torch.nn.ReLU(),
#    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # 32x22x22
#    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0, stride=1), # 64x18x18
#    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # 64x9x9
#    torch.nn.Flatten(),                                                                   # 5184 features
#    torch.nn.Linear(in_features=64*9*9, out_features=1000),                                # 1000 features
#    torch.nn.ReLU(),
#    torch.nn.Linear(in_features=1000, out_features=len(transfer_labels)))                 # Target classes

# Input is a 56x56 image
#tiny = torch.nn.Sequential(
#    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=0, stride=1),  # 32x52x52
#    torch.nn.BatchNorm2d(32),
#    torch.nn.ReLU(),
#    torch.nn.Dropout2d(p=0.5),
#    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),  # 32x52x52
#    torch.nn.BatchNorm2d(32),
#    torch.nn.ReLU(),
#    torch.nn.Dropout2d(p=0.5),
#    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # 32x26x26
#    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0, stride=1), # 32x22x22
#    torch.nn.BatchNorm2d(32),
#    torch.nn.ReLU(),
#    torch.nn.Dropout2d(p=0.5),
#    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # 32x11x11
#    torch.nn.Flatten(),                                                                   # 3872 features
#    torch.nn.Linear(in_features=32*11*11, out_features=1000),                             # 1000 features
#    torch.nn.ReLU(),
#    torch.nn.Linear(in_features=1000, out_features=100),                                  # 100 features
#    torch.nn.ReLU(),
#    torch.nn.Linear(in_features=100, out_features=len(transfer_labels)))                  # Target classes

# Input is a 56x56 image
tiny = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=0, stride=1),  # 32x52x52
    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # 32x26x26
    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0, stride=1), # 32x22x22
    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                               # 32x11x11
    torch.nn.Flatten(),                                                                   # 3872 features
    torch.nn.Linear(in_features=32*11*11, out_features=len(transfer_labels)))                               # Target classes

tiny.train()

#optimizer = torch.optim.SGD(tiny.parameters(), lr=0.05, momentum=0.001, weight_decay=0.00)
optimizer = torch.optim.Adagrad(tiny.parameters(), lr=0.01, weight_decay=0.001)
# We use L1 because the desired outputs are relative to the other DNN's outputs.
# We don't want to penalize our small DNN for a single output's mismatch, only
# the relative scores of the outputs
loss_fn = torch.nn.L1Loss()


def run_demo():
    """Demo data collection, object detection, and tracking with a simple GUI."""
    parser = argparse.ArgumentParser(
        description="Transfer learning demo")
    parser.add_argument(
        '--class_model',
        type=str,
        default='image_classification_ppresnet50_2022jan.onnx',
        help='Path to the classifying model.')
    args = parser.parse_args()

    # Initialize the classifier.
    # It has fewer convenience wrappers, so we made our own.
    vectorizer = PPResnet(args.class_model)

    # Some display constants
    text_yellow=(0, 255, 255)
    text_blue=(255, 0, 0)
    text_green=(0, 255, 0)
    text_red=(0, 0, 255)

    # Open the webcam
    video = cv2.VideoCapture(0)

    # Check and set some properties
    # See the OpenCV docs for a list of properties:
    # https://docs.opencv.org/4.11.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    # or samples in OpenCV repository: python/samples/video.py and python/samples/video_v4l2.py
    vid_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Sample bboxes will be 224 pixels wide. We want to always be able to fit two 224x224 boxes in an image,
    # so set a minimum frame size of 960x720 pixels
    if vid_height < 720:
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vid_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Some parts of this are slow, but we only want the latest frame. Set things to drop older frames by reducing the buffer size.
    video.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # Remember the current UI state.
    # waiting: the user needs to click a button to begin training
    # training: the user is training
    # classifying: the new model is classifying
    # quitting: the user wants to exit
    mode = "waiting"

    # OpenCV's waitKey function returns not just the key, but also any keyboard modifiers. This
    # means that the returned value cannot be compared to just the key.
    def isKey(cvkey, key):
        return key == (cvkey & 0xFF)

    # Keep going until capture fails or the user quits
    has_frame, frame = video.read()

    training_images = []
    training_labels = []

    if not has_frame:
        print("Capture failures, exiting!")
        exit(1)

    # Some local functions to bounding box logic
    def centerToBox(x, y, width):
        safe_x = min(vid_width - width//2 - 1, max(x, width//2))
        safe_y = min(vid_height - width//2 - 1, max(y, width//2))
        return safe_x - width//2, safe_y - width//2, safe_x + width//2, safe_y + width//2

    while mode != "quitting" and has_frame:
        display_frame = frame.copy()
        #frame_h, frame_w, _ = display_frame.shape
        if mode == "waiting":
            cv2.putText(display_frame, "Press any button to begin data collection.", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)
        elif mode == "training":
            # Train from boxes of different sizes to get more variety
            boxes = [
                centerToBox(vid_width//2, vid_height//2, int(vid_height*0.9)),
                centerToBox(vid_width//2, vid_height//2, int(vid_height*0.8)),
                centerToBox(vid_width//2, vid_height//2, vid_width//2),
                centerToBox(vid_width//2, vid_height//2, vid_width//3),
                centerToBox(vid_width//2, vid_height//2, vid_width//5)]
            windows = [frame[top:bottom, left:right] for left, top, right, bottom in boxes]
            # TODO Run in a batch rather than one at a time
            class_scores = [vectorizer.getClassScores(window)[0][0] for window in windows]
            # Only train with the class scores that we care about
            for idx, scores in enumerate(class_scores):
                # Our model will consume 56x56 images
                # But don't sample if one of our target classes isn't in the top 5
                top_classes = numpy.argsort(scores)[-5:]
                targets_in_top = [imagenet_labels[cidx] in transfer_labels for cidx in top_classes]
                draw_color = text_red
                if any(targets_in_top):
                    training_images.append(tiny_preprocess(windows[idx]))
                    new_labels = torch.nn.functional.softmax(torch.tensor([scores[imagenet_labels.index(name)] for name in transfer_labels]), dim=0)
                    training_labels.append(new_labels)
                    draw_color = text_green

                # Draw the prediction box and prediction so the user knows what's happening
                left, top, right, bottom = boxes[idx]
                cv2.rectangle(display_frame, (left, top), (right, bottom), draw_color, 1)
                cv2.putText(display_frame, f"Prediction: {imagenet_labels[top_classes[-1]]}({scores[top_classes[-1]]})", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color)
            cv2.putText(display_frame, "Press any button to end data collection.", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)
        elif mode == "classifying":
            # Classify whatever is in the middle of the image
            search_center = (vid_width//2, vid_height//2)

            box = centerToBox(vid_width//2, vid_height//2, vid_width//2)
            left, top, right, bottom = box
            window = frame[top:bottom, left:right]
            # Run the new model. Make sure we aren't capturing a gradient and running out of memory.
            with torch.no_grad():
                processed_image = torch.tensor(numpy.array([tiny_preprocess(window)])).float()
                output = tiny(processed_image)[0]
                ranked_scores = torch.argsort(output, descending=True)
            top_score = scores[ranked_scores[0]].item()
            top_class = transfer_labels[ranked_scores[0]]

            # Draw the prediction box and prediction
            cv2.rectangle(display_frame, (left, top), (right, bottom), text_green, 1)
            cv2.putText(display_frame, f"Prediction: {top_class}({top_score})", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, text_yellow)

        cv2.imshow('Transfer Demo', display_frame)

        key = cv2.waitKey(1)
        if key > 0:
            if isKey(key, ord('q')):
                print("Quitting")
                mode = "quitting"
            elif mode == "waiting":
                mode = "training"
            elif mode == "training":
                # Train the model
                # Shuffle the data first
                training_order = list(range(len(training_images)))
                random.shuffle(training_order)
                batch_size = 16
                total_batches = len(training_order)//batch_size
                if len(training_order) % batch_size > 0:
                    total_batches = total_batches + 1
                print(training_order)
                print(f"Training with {len(training_images)} samples over {total_batches} batches")
                for epoch in range(20):
                    total_loss = 0
                    for i in range(total_batches):
                        indices = training_order[i*batch_size:(i+1)*batch_size]
                        batch_images = torch.tensor(numpy.array([training_images[index] for index in indices])).float()
                        batch_labels = torch.tensor(numpy.array([training_labels[index] for index in indices]))

                        # Zero the gradients, forward the batch, compute the loss, backpropagate,
                        # and update the parameters.
                        optimizer.zero_grad()
                        out = tiny(batch_images)
                        loss = loss_fn(out, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss = total_loss + loss.sum().item()
                    print(f"Epoch {epoch} average loss {total_loss/len(training_images)}")

                # Leave training mode
                tiny.eval()

                mode = "classifying"

        # Grab the next frame
        has_frame, frame = video.read()

    # Cleanup
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_demo()
