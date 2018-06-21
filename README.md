# Face Swap
A set of tools and utils to operate face swapping and explore creative usage of derived artifacts.

## Setup

Install via

    python setup.py [develop]

Notice that the project depends on [another repository of mine](https://github.com/5agado/data-science-learning) for reusable utils and modules. This dependency should be installed via the previous command, but if you encounter any problem you can run

```
git clone https://github.com/5agado/data-science-learning.git
cd data-science-learning
pip install .
```

Configuration settings and models location should be specified in *models.cfg*.

## Run

### Scripts
*scripts.sh* provides a simplified interface to various operations and setup based on video conversion. Requires a workdir containing a video file (i.e. .gif, .webm, mp4) and requires [ffmpeg](https://www.ffmpeg.org/) to be installed on the machine.

Init dir (required step). Creates required folders

    ./scripts.sh -w ~/workdir/path -i

Extract faces (relies on params set in *models.cfg*)

    ./scripts.sh -w ~/workdir/path -e

Swap faces and create video (relies on params set in *models.cfg*)

    ./scripts.sh -w ~/workdir/path -s

### Python Commands
As an alternative to the calling scripts you can rely directly on Python commands installed during initial setup.

Faces Extract

    extract -i dir/to/video -o output/dir/faces

Faces Swap

    deep_swap -i dir/to/video -o output/video/path -A -model_name base_gan -model_version v0

## Train
You can train a new model by using *train.py*. See `train.py -h` for more info.

Otherwise I would suggest to rely on [Google Colaboratory](https://colab.research.google.com/), which provides a free GPU environment accessible via Jupyter. See [my related notebook](notebooks/Faceswap_Train_GAN.ipynb) which installs all the required dependencies and then provides basic interface to the training functions.

# Resources
This code is colossally based on [shaoanlu repo and work](https://github.com/shaoanlu/faceswap-GAN). I tried to give it a bit more structure, and add comments and pointers for new comers to better understand what exactly is going on.

Other useful references for code and techniques for pre and post-processing phases:

* [Face Swap using OpenCV](https://www.learnopencv.com/face-swap-using-opencv-c-python/)
* [How to install dlib](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)
* [Detect eyes, nose, lips, and jaw with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/)
* [Face Alignment with OpenCV and Python](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)

# Docker
You can also build a Docker image (complete of all requirement code and utils) via

    docker build -t faceswap .

Start image and exec bash

    docker run -it faceswap bash

# TODO
* align face -> feed to generator -> detect face in generated image -> align back using landmarks
* try 128 input (also good to test if all methods are generic)

# License

Released under version 2.0 of the [Apache License].

[Apache license]: http://www.apache.org/licenses/LICENSE-2.0