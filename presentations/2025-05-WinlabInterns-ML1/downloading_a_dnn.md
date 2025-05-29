---
title: So You've Downloaded a DNN, Now What?
---

<!--
Abstract:

It's easy to get started with neural networks -- you can just download one and try it out!
From YOLO to Stable Diffusion, downloading and running a neural network locally is now something you can get working before lunch.
But what if you don't just want to run a model; what if you want good results?
-->

# So You've Downloaded a DNN

## Now What?

Bernhard Firner

2025-05-27

---

## Pretrained Models

* First, what does pretrained mean?
* It means you don't need to train the model
  * A process involving large amounts of data, time, and cursing
* The end result is a bunch of numbers
  * The numbers are the parameters for some linear algebra
  * In the end, a DNN is just a *lot* of math

---

## Feeding Your Square Data Into a Round Model
* Downloading a model is easy; using it can be hard
* Your data (images, etc) may still need to be formatted
  * Cropping, scaling
  * Normalizing colors, converting to RGB, YUV, grayscale, etc
  * Selecting frames from a video
  * Selecting detection thresholds for your data
  * So many details

---

## Don't Download Whatever!

* Some models may be harder to get working!
* The latest and greatest models also stand on their own
  * Try to get access to more models with less work!
* As an example, let's say you search for YOLO (You Only Look Once)

---

## Latest and Greatest

Top result on the day I searched\
[https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

<img src="./figures/2025-05-27_16-41-UltralyticsYolo.png" style="height: 750px" />

---

## More Modular

MMDetection is a group of many detection DNNs\
[https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

<img src="./figures/2025-05-27_16-47-MMYolo.png" style="height: 750px" />

---

## Easiest to Use

Some DNNs are already integrated into OpenCV\
[https://github.com/opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)

<img src="./figures/2025-05-27_16-55-OpenCVZoo.png" style="height: 750px" />

---

## What is OpenCV?

* Research quality computer vision toolset
  * Good for prototyping and experimenting
* Where you should start


---

## But I Want the Super Latest Most New Model!!!!!

* No you don't

---

## Metrics

* YOLO does detection
* How well does it detect?
  * A combination of recall and precision
  * recall: ${True Positives} / {All Instances}$
  * precision: ${True Positives} / {All Positives}$
* Recall and precision are calculated per class

---

## Latest and Greatest

* A paper is published when a model improves mAP by 1%
  * You probably don't even know what mAP is
  * $MAP = \sum^Q_{q=1}AveP(q) / Q$
    * AveP (Average Precision) is the area under the precision-recall curve
  * See [mAP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
* Real world results won't change
  * You'll have to deal with model errors either way

---

## Python Setup

<section style="text-align: left;">

> `pip3 install opencv-python`
* The `opencv-python` package will install the OpenCV binaries
* See the [package description](https://pypi.org/project/opencv-python)

---

## C++ Setup

[https://github.com/opencv/opencv](https://github.com/opencv/opencv)
* Follow the [OpenCV install docs](https://docs.opencv.org/4.x/d0/d3d/tutorial_general_install.html)

---

## Model Zoo

* Clone the model zoo [https://github.com/opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)
* There is no installation, the models are static files
* For example, YOLOX models are in opencv_zoo/models/object_detection_yolox
  * object_detection_yolox_2022nov.onnx
* ONNX stands for "Open Neural Network eXchange"
  * Standard way to share model parameters
---

## Python Installation (Quick Version)

<section style="text-align: left;">

> pip3 install opencv-python
> git clone https://github.com/opencv/opencv_zoo
> cd opencv_zoo
> git lfs install
> git lfs pull

---

## Detection Example

See [opencv_zoo/models/object_detection_yolox/yolox.py](https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/yolox.py)
```python
import numpy as np
import cv2

class YoloX:
    def __init__(self, modelPath, confThreshold=0.35, nmsThreshold=0.5, objThreshold=0.5, backendId=0, targetId=0):
        self.num_classes = 80
        self.net = cv2.dnn.readNet(modelPath)
        self.input_size = (640, 640)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.strides = [8, 16, 32]
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.backendId = backendId
        self.targetId = targetId
        self.net.setPreferableBackend(self.backendId)
        self.net.setPreferableTarget(self.targetId)

        self.generateAnchors()

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self.backendId = backendId
        self.targetId = targetId
        self.net.setPreferableBackend(self.backendId)
        self.net.setPreferableTarget(self.targetId)

    def preprocess(self, img):
        blob = np.transpose(img, (2, 0, 1))
        return blob[np.newaxis, :, :, :]

    def infer(self, srcimg):
        input_blob = self.preprocess(srcimg)

        self.net.setInput(input_blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        predictions = self.postprocess(outs[0])
        return predictions

    def postprocess(self, outputs):
        dets = outputs[0]

        dets[:, :2] = (dets[:, :2] + self.grids) * self.expanded_strides
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * self.expanded_strides

        # get boxes
        boxes = dets[:, :4]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

        # get scores and class indices
        scores = dets[:, 4:5] * dets[:, 5:]
        max_scores = np.amax(scores, axis=1)
        max_scores_idx = np.argmax(scores, axis=1)

        keep = cv2.dnn.NMSBoxesBatched(boxes_xyxy.tolist(), max_scores.tolist(), max_scores_idx.tolist(), self.confThreshold, self.nmsThreshold)

        candidates = np.concatenate([boxes_xyxy, max_scores[:, None], max_scores_idx[:, None]], axis=1)
        if len(keep) == 0:
            return np.array([])
        return candidates[keep]

    def generateAnchors(self):
        self.grids = []
        self.expanded_strides = []
        hsizes = [self.input_size[0] // stride for stride in self.strides]
        wsizes = [self.input_size[1] // stride for stride in self.strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
            xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            self.grids.append(grid)
            shape = grid.shape[:2]
            self.expanded_strides.append(np.full((*shape, 1), stride))

        self.grids = np.concatenate(self.grids, 1)
        self.expanded_strides = np.concatenate(self.expanded_strides, 1)
```


---

## Detection Example

See [opencv_zoo/models/object_detection_yolox/demo.py](https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/demo.py)
```python
import numpy as np
import cv2 as cv
import argparse

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from yolox import YoloX

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def letterbox(srcimg, target_size=(640, 640)):
    padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
    ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
    resized_img = cv.resize(
        srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img

    return padded_img, ratio

def unletterbox(bbox, letterbox_scale):
    return bbox / letterbox_scale

def vis(dets, srcimg, letterbox_scale, fps=None):
    res_img = srcimg.copy()

    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(res_img, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for det in dets:
        box = unletterbox(det[:4], letterbox_scale).astype(np.int32)
        score = det[-2]
        cls_id = int(det[-1])

        x0, y0, x1, y1 = box

        text = '{}:{:.1f}%'.format(classes[cls_id], score * 100)
        font = cv.FONT_HERSHEY_SIMPLEX
        txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
        cv.rectangle(res_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv.rectangle(res_img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), (255, 255, 255), -1)
        cv.putText(res_img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 0, 0), thickness=1)

    return res_img

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--model', '-m', type=str, default='object_detection_yolox_2022nov.onnx',
                        help="Path to the model")
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
    parser.add_argument('--confidence', default=0.5, type=float,
                        help='Class confidence')
    parser.add_argument('--nms', default=0.5, type=float,
                        help='Enter nms IOU threshold')
    parser.add_argument('--obj', default=0.5, type=float,
                        help='Enter object threshold')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Specify to save results. This flag is invalid when using camera.')
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Specify to open a window for result visualization. This flag is invalid when using camera.')
    args = parser.parse_args()

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    model_net = YoloX(modelPath= args.model,
                      confThreshold=args.confidence,
                      nmsThreshold=args.nms,
                      objThreshold=args.obj,
                      backendId=backend_id,
                      targetId=target_id)

    tm = cv.TickMeter()
    tm.reset()
    if args.input is not None:
        image = cv.imread(args.input)
        input_blob = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        input_blob, letterbox_scale = letterbox(input_blob)

        # Inference
        tm.start()
        preds = model_net.infer(input_blob)
        tm.stop()
        print("Inference time: {:.2f} ms".format(tm.getTimeMilli()))

        img = vis(preds, image, letterbox_scale)

        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', img)

        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, img)
            cv.waitKey(0)

    else:
        print("Press any key to stop video capture")
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            input_blob, letterbox_scale = letterbox(input_blob)

            # Inference
            tm.start()
            preds = model_net.infer(input_blob)
            tm.stop()

            img = vis(preds, frame, letterbox_scale, fps=tm.getFPS())

            cv.imshow("YoloX Demo", img)

            tm.reset()
```

---

## Segmentation Example

* With EfficientSAM
* Also adding some UI

---

## Tracking Example

* With VitTrack
