# Create /tmp/.docker.xauth if it does not already exist.
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    touch $XAUTH
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
fi

xhost +
docker run \
    `# Share the host’s network stack and interfaces. Allows multiple containers to interact with each other.` \
    --ipc=host \
    `# Interactive processes, like a shell.` \
    -it \
    `# Clean up the container after exit.` \
    --rm \
    `# Use GUI and NVIDIA.` \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="/dev/dri:/dev/dri" \
    --gpus all \
    `# Mount the folders.` \
    -v ${PWD}:/home/ws \
    yolo_limo