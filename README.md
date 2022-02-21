# TacoTrashDetection
This project uses the [TACO](http://tacodataset.org/) (Trash Annotations in Context) [dataset](https://www.kaggle.com/kneroma/tacotrashdataset) and a [pre-trained Tensorflow model](https://www.kaggle.com/bouweceunen/training-ssd-mobilenet-v2-with-taco-dataset) to detect trash from images with Python.

# Installation
This project is really meant as a demonstration. I do not actually recommend using it. It is completely up to you
Prerequisites:
- Matplotlib
- Numpy
- Pandas
- OpenCV (for live detection)
- [Tensorflow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

# Usage
Run the taco.py file and provide an image as "input.png", and the result will be in the "output.png" file. Example:

```
python .\taco.py
Building label map from examples
Label map witten to labelmap.pbtxt
Reconstructing Tensorflow model
Success!
Using single image detection. Processing...
Finished. Image saved to file
Elapsed: 5.49 seconds
```
# Example Image
![Example Image](/output.png)

# Legit License
Copyright 2022 <Vlad Chira>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
