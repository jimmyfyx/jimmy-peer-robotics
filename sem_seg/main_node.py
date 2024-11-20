import rclpy
import cv2
import sys
import os
import torch
import numpy as np

from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, LivelinessPolicy
from torchvision import transforms

from .dinoVit import dinovit


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('seg_node')
        self.bridge = CvBridge()

        # Parse arguments
        custom_args = sys.argv[1:]
        h = int(custom_args[0])
        w = int(custom_args[1])
        rgb_topic = custom_args[2]
        cam_info_topic = custom_args[3]
        
        # Subscribers and Publishers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            durability=DurabilityPolicy.VOLATILE,      
            liveliness=LivelinessPolicy.AUTOMATIC,     
            depth=10                                   
        )

        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, qos_profile)
        self.cam_info_sub = self.create_subscription(CameraInfo, cam_info_topic, self.cam_info_callback, qos_profile)

        self.det_pub = self.create_publisher(Image, '/detection/image', 10)
        self.seg_pub = self.create_publisher(Image, '/segmentation/image', 10)

        # Variables for storing image and camera info
        self.rgb_image = None
        self.cam_info_msg = None

        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seg_model = dinovit.DINOv2DPT(output_size=(h, w)).to(self.device)
        self.seg_model.decoder.load_state_dict(torch.load('./dinoVit/checkpoints/decoder/epoch20_0.047767_0.130500.pth'))
        self.seg_model.final_layer.load_state_dict(torch.load('./dinoVit/checkpoints/final_layer/epoch20_0.047767_0.130500.pth'))
        self.det_model = YOLO("./yolo11s_pallet.pt")

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])

        print('Node Initialized!')

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)  # BGR to RGB

            # Object detection
            results = self.det_model(self.rgb_image)
            annotated_frame = results[0].plot()

            # Semantic Segmentation
            rgb_image_resized = cv2.resize(self.rgb_image, (420, 420), interpolation=cv2.INTER_LINEAR)
            rgb_tensor = self.transform_rgb(rgb_image_resized).to(self.device)  # (3, 420, 420)
            rgb_tensor = rgb_tensor.unsqueeze(0)
            with torch.no_grad():
                masks = self.seg_model(rgb_tensor)
            masks = masks.squeeze(0).cpu().numpy()

            # Overlay the masks on RGB image
            colored_mask = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.float32)
            colored_mask[masks[0] > 0.5, 0] = 255.0  # Red for ground
            colored_mask[masks[1] > 0.5, 1] = 255.0  # Green for pallet
            overlay = cv2.addWeighted(self.rgb_image.astype(np.float32), 0.7, colored_mask, 0.3, 0).astype(np.uint8)

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)  # RGB to BGR for visualization
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            det_img_msg = self.bridge.cv2_to_imgmsg(annotated_frame)
            seg_img_msg = self.bridge.cv2_to_imgmsg(overlay)
            self.det_pub.publish(det_img_msg)
            self.seg_pub.publish(seg_img_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to process RGB image: {e}")

    def cam_info_callback(self, msg):
        self.cam_info_msg = msg
        

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
