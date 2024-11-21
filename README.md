# jimmy-peer-robotics

This README only includes instructions to run the ROS2 node. For implementations of the overall pipeline, evaluations, and potential improvements, I document them in a separate [Notion page](https://bitter-taxicab-796.notion.site/Peer-Robotics-Vision-Assignment-d2e2020a2f07415a95d90304c1a5f9aa). Hope this helps!

The node has been fully tested on my workstation, which has:
- Ubuntu 20.04
- ROS2 Foxy
- NVIDIA driver version 535.183.01
- CUDA version 12.2
- *Graphic card:* NVIDIA GeForce RTX 3060 Mobile (6GB RAM)


## **Clone the Repository and Build Workspace**

- Go the the root directory of a ROS2 workspace.
- Clone this repository into the `/src` directory:

   ```bash
   cd /src
   git clone https://github.com/jimmyfyx/jimmy-peer-robotics.git sem_seg
   ```
- Build the workspace:
    ```bash
    cd ..
    source /opt/ros/<ros-distro>/setup.bash
    colcon build --packages-select sem_seg --symlink_install
    ```
## **Environment**
I created a CONDA environment to run the ROS2 node, which should only depend on three main packages: `torch`, `opencv-python`, and `ultralytics`. 
- Create the conda environment with `python3.8`:
    ```bash
    conda create --name semseg python=3.8
    ```
- Install the packages:
    ```bash
    conda activate semseg
    (semseg) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    (semseg) pip install opencv-python==4.10.0.84
    (semseg) pip install ultralytics
    ```
Specifically, PyTorch 2.0 or above is recommended as the code requires the usage of `torch.hub` to download pretrained weights. Check CUDA availability for PyTorch after installing.

## **Run the Node**
- In the root directory of the workspace, activate the CONDA environment and source the workspace:
    ```bash
    conda activate semseg
    (semseg) source /opt/ros/<ros-distro>/setup.bash
    (semseg) source install/setup.bash
    ``` 

- Before running the node, we need to make sure ROS2 can interpret the correct `$PYTHONPATH` to access the required dependecies. Check the path:
    ```bash
    (semseg) $PYTHONPATH
    ``` 
  If the output doesn't include paths to python packages installed within CONDA (etc. `/home/.../anaconda3/.../lib/python3.8/site-packages`), add the path:
    ```bash
    (semseg) export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
    ``` 
- Now we're ready to run the node. From the root directory of the workspace, navigate to:
    ```bash
    (semseg) cd /src/sem_seg/sem_seg
    ```
- Run the node:
    ```bash
    (semseg) ros2 run sem_seg seg_node <img_h> <img_w> <rgb_img_topic> <cam_info_topic>
    ```
    - `img_h`: The height of the received RGB image.
    - `img_w`: The width of the received RGB image.
    - `rgb_img_topic`: The RGB image topic name.
    - `cam_info_topic`: The camera info topic name. If no camera info topic is being published, fill it with **None**.  

**Notes**:
- The node class currently has some specific QoS settings, adjust them if needed.

## **Outputs**
Currently, the node publishes two topics:
- `/detection/image`: The RGB image showing YOLO detection bounding boxes.
- `/segmentation/image`: The RGB image showing semantic segmentation results. Green and red masks represent pallet and ground respectively.