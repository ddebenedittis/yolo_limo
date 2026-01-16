# YOLO Limo

Track the motion of Limo robots in a video and display their path.

## Labelme Setup

To label the Limo robots in your videos, you can use Labelme.

Create a virtual environment, install labelme and labelme2yolo:
```bash
python3 -m venv .venv
pip3 install -r requirements.txt
```

## Docker

Install Docker following [this guide](https://github.com/ddebenedittis/docker_ros_nvidia?tab=readme-ov-file#usage).

Build the Docker image with
```bash
./docker/build.bash
```

and run it with
```bash
./docker/run.bash
```

## Usage

Train the YOLO model with:
```bash
python3 src/train.py
```

Visualize the inference results on a video with:
```bash
python3 src/detect.py
```