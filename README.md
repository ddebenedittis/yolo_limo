# YOLO Limo

Track the motion of Limo robots in a video and display their path.

## Labelme Setup

To label the Limo robots in your videos, you can use Labelme.

Create a virtual environment, install labelme and labelme2yolo:
```bash
python3 -m venv .venv
pip3 install -r requirements.txt
```

## Labelme Usage

Enter in the .venv and run labelme with
```shell
labelme
```

Label your dataset. Once done, run
```shell
labelme2yolo --json_dir ./img --val_size 0.20
```
to export the dataset in a Yolo-compatible format.

Move the dataset in the datasets folder.

The dataset.yaml file will be like this:
```yaml
train: ./images/train
val: ./images/val
test:

names:
    1: limo
    0: limo_arm
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