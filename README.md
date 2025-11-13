
---

# Omni3D with Custom Object Tracker

**Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild**  
This repository extends the Omni3D model by integrating a custom object tracking mechanism, enhancing 3D detection with continuous tracking across frames.

## Table of Contents:
1. [Overview](#overview)
2. [Installation](#installation)
3. [Running the Demo](#demo)
4. [Training](#training)
5. [Inference](#inference)
6. [Tracker Implementation](#tracker)
7. [Chat with Phi 3 Vision](#chatwithphi3vision)
8. [Citing Omni3D](#citing)
9. [License](#license)
10. [Contributing](#contributing)


## Overview <a name="overview"></a>

Omni3D, originally developed by [Garrick Brazil et al.](https://garrickbrazil.com/omni3d), is a state-of-the-art model for 3D object detection. This project incorporates a custom tracking mechanism to extend the detection capabilities, enabling real-time object tracking in various environments.

For more details on the Omni3D project, refer to the [original repository](https://garrickbrazil.com/omni3d).

## Installation <a name="installation"></a>

Follow the steps below to set up the environment:

```bash
# Create and activate a new conda environment
conda create -n cubercnn python=3.8
source activate cubercnn

# Install main dependencies
conda install -c fvcore -c iopath -c conda-forge -c pytorch3d -c pytorch fvcore iopath pytorch3d pytorch=1.8 torchvision=0.9.1 cudatoolkit=10.1

# Install additional dependencies
pip install cython opencv-python
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
conda install -c conda-forge scipy seaborn
```

## Running the Demo <a name="demo"></a>

```bash
# Download sample images
sh demo/download_demo_COCO_images.sh

# Run the demo with the custom tracker
python demo/demo_detection.py \
--config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
--input-folder "datasets/coco_examples" \
--threshold 0.25 --display \
MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
OUTPUT_DIR output/demo_with_tracking
```

## Running the Demo with Tracker <a name="demo"></a>

```bash
# Download sample images
sh demo/download_demo_COCO_images.sh

# Run the demo with the custom tracker
python demo/demo_tracker.py \
--config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
--input-video "demo/video_indoor2.mp4" \
--threshold 0.40 --display 
```

## Training <a name="training"></a>

To train the Omni3D model with tracking:

```bash
python tools/train_net.py \
  --config-file configs/Base_Omni3D.yaml \
  OUTPUT_DIR output/omni3d_with_tracking
```

## Inference <a name="inference"></a>

```bash
python tools/train_net.py \
  --eval-only --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
  MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
  OUTPUT_DIR output/evaluation
```

## Tracker Implementation <a name="tracker"></a>

This project adds an object tracker to the original Omni3D model. The tracker matches detected objects across frames using a custom algorithm based on:
- 3D bounding box information
- GIoU and Iou 3D computation
- Object centers
- Category types
- Chamfer Distance 

Key features:
- Custom matching logic for continuous tracking across frames
- High-cost match handling for challenging detections
  
![Progetto senza titolo (1) (1)](https://github.com/user-attachments/assets/44635208-52d4-4780-ab61-9b2a3705ce11)
![Screenshot from 2024-07-23 16-27-00 (1)](https://github.com/user-attachments/assets/859a057a-56e8-41e3-ab53-13231e94e3cf)

## Citing Omni3D <a name="citing"></a>

Please cite the original Omni3D paper:

```BibTeX
@inproceedings{brazil2023omni3d,
  author =       {Garrick Brazil and Abhinav Kumar and Julian Straub and Nikhila Ravi and Justin Johnson and Georgia Gkioxari},
  title =        {{Omni3D}: A Large Benchmark and Model for {3D} Object Detection in the Wild},
  booktitle =    {CVPR},
  address =      {Vancouver, Canada},
  month =        {June},
  year =         {2023},
  organization = {IEEE},
}
```

---

# Chat with Phi-3 Vision <a name="chatwithphi3vision"></a>

![CHAT4](https://github.com/user-attachments/assets/68f3b64e-a070-49e6-b813-582354134085)


This folder is part of the **Monocular 3D Object Detection and Tracking** project and contains a Streamlit-based application that allows users to interact with a vision model, specifically Phi-3 Vision, to analyze images and return detailed descriptions.

## Overview

The application is designed to:
- Receive images from a server via a socket connection
- Allow users to submit specific queries about the images
- Leverage Phi-3 Vision to provide detailed descriptions of image content

Key features:
- **Real-time communication** with a server for image acquisition
- **Text-based interaction** for specific image details
- **Automatic frame fetching** upon query submission
- **Customizable chat interface** that resets after each response
- **Responsive design** for smooth user experience

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ramonatarantino/mocular-3d-object-detection-tracking.git
   cd mocular-3d-object-detection-tracking/chat-with-phi-3-vision
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Start the server:**
   Ensure the server providing the image stream is running.

## License (Chat with Phi 3 Vision) <a name="license"></a>

This project is licensed under the MIT License.  
The server code is based on the work from [this repository](https://github.com/bhimrazy/chat-with-phi-3-vision).

## Contributing <a name="contributing"></a>

Contributions are welcome! Feel free to open issues or submit pull requests.

---
