import base64
import socket
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from smartfactory_interfaces.msg import YoloPose
from ultralytics import YOLO

POSE_TOPIC = "/jetcobot/storage/camera/yolo_pose"
RAW_DISPLAY_WINDOW_NAME = "YOLO UDP Pose (Raw)"
FILTERED_DISPLAY_WINDOW_NAME = "YOLO UDP Pose (Filtered)"


def load_intrinsics(calib_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    calib = np.load(str(calib_path))
    camera_matrix = calib["camera_matrix"].astype(np.float64)
    dist_coeffs = calib["dist_coeffs"].astype(np.float64)

    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(-1, 1)
    elif dist_coeffs.ndim == 2 and dist_coeffs.shape[0] == 1:
        dist_coeffs = dist_coeffs.reshape(-1, 1)

    return camera_matrix, dist_coeffs


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected (4,2), got {pts.shape}")

    rect = np.zeros((4, 2), dtype=np.float64)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_detection_candidates(
    result_obj: Any, conf_thres: float
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    names = getattr(result_obj, "names", {})

    if getattr(result_obj, "obb", None) is not None and len(result_obj.obb) > 0:
        obb_xyxyxyxy = result_obj.obb.xyxyxyxy.cpu().numpy()
        confs = (
            result_obj.obb.conf.cpu().numpy()
            if getattr(result_obj.obb, "conf", None) is not None
            else np.ones((len(obb_xyxyxyxy),), dtype=np.float64)
        )
        clses = (
            result_obj.obb.cls.cpu().numpy().astype(int)
            if getattr(result_obj.obb, "cls", None) is not None
            else np.full((len(obb_xyxyxyxy),), -1, dtype=int)
        )

        for det_idx, (pts, conf, cls_id) in enumerate(
            zip(obb_xyxyxyxy, confs, clses)
        ):
            if float(conf) < conf_thres:
                continue
            ordered = order_points_tl_tr_br_bl(pts.reshape(4, 2))
            label = (
                names.get(int(cls_id), str(int(cls_id)))
                if isinstance(names, dict) and cls_id >= 0
                else "unknown"
            )
            candidates.append(
                {
                    "det_idx": int(det_idx),
                    "mode": "obb",
                    "conf": float(conf),
                    "cls_id": int(cls_id),
                    "label": label,
                    "img_points": ordered,
                }
            )
        return candidates

    if getattr(result_obj, "boxes", None) is not None and len(result_obj.boxes) > 0:
        xyxy = result_obj.boxes.xyxy.cpu().numpy()
        confs = (
            result_obj.boxes.conf.cpu().numpy()
            if getattr(result_obj.boxes, "conf", None) is not None
            else np.ones((len(xyxy),), dtype=np.float64)
        )
        clses = (
            result_obj.boxes.cls.cpu().numpy().astype(int)
            if getattr(result_obj.boxes, "cls", None) is not None
            else np.full((len(xyxy),), -1, dtype=int)
        )

        for det_idx, (bbox, conf, cls_id) in enumerate(zip(xyxy, confs, clses)):
            if float(conf) < conf_thres:
                continue
            x1, y1, x2, y2 = bbox.astype(np.float64)
            pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
            )
            ordered = order_points_tl_tr_br_bl(pts)
            label = (
                names.get(int(cls_id), str(int(cls_id)))
                if isinstance(names, dict) and cls_id >= 0
                else "unknown"
            )
            candidates.append(
                {
                    "det_idx": int(det_idx),
                    "mode": "bbox",
                    "conf": float(conf),
                    "cls_id": int(cls_id),
                    "label": label,
                    "img_points": ordered,
                }
            )

    return candidates


def estimate_poses(
    detections: list[dict[str, Any]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    object_width_m: float,
    object_height_m: float,
) -> list[dict[str, Any]]:
    object_points = np.array(
        [
            [-object_width_m / 2, -object_height_m / 2, 0.0],
            [object_width_m / 2, -object_height_m / 2, 0.0],
            [object_width_m / 2, object_height_m / 2, 0.0],
            [-object_width_m / 2, object_height_m / 2, 0.0],
        ],
        dtype=np.float64,
    )

    poses: list[dict[str, Any]] = []
    for det in detections:
        image_points = det["img_points"].astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue

        tx, ty, tz = tvec.reshape(3)
        poses.append(
            {
                **det,
                "rvec": rvec,
                "tvec": tvec,
                "xyz": (float(tx), float(ty), float(tz)),
            }
        )
    return poses


def rotation_matrix_to_quaternion(
    rot: np.ndarray,
) -> tuple[float, float, float, float]:
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rot[2, 1] - rot[1, 2]) * s
        qy = (rot[0, 2] - rot[2, 0]) * s
        qz = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat /= norm
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def rvec_to_quaternion(rvec: np.ndarray) -> tuple[float, float, float, float]:
    rot, _ = cv2.Rodrigues(rvec)
    return rotation_matrix_to_quaternion(rot)


def quaternion_to_rvec(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Invalid quaternion norm")
    q = q / norm
    x, y, z, w = q

    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(rot)
    return rvec


def average_quaternions(quaternions: list[np.ndarray]) -> np.ndarray:
    if not quaternions:
        raise ValueError("Quaternion list is empty")

    aligned = []
    ref = np.asarray(quaternions[0], dtype=np.float64).reshape(4)
    for q in quaternions:
        qn = np.asarray(q, dtype=np.float64).reshape(4)
        if np.dot(ref, qn) < 0.0:
            qn = -qn
        aligned.append(qn)

    mean_q = np.mean(np.stack(aligned, axis=0), axis=0)
    norm = np.linalg.norm(mean_q)
    if norm <= 0.0:
        return ref / np.linalg.norm(ref)
    return mean_q / norm


def draw_pose_overlay(
    frame: np.ndarray,
    poses: list[dict[str, Any]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
) -> np.ndarray:
    display = frame.copy()

    for pose in poses:
        img_points = pose["img_points"].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            display,
            [img_points],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
        )
        cv2.drawFrameAxes(
            display,
            camera_matrix,
            dist_coeffs,
            pose["rvec"],
            pose["tvec"],
            axis_length_m,
            2,
        )

        x, y, z = pose["xyz"]
        label = f'{pose["label"]} x={x:.3f} y={y:.3f} z={z:.3f}'
        text_org = tuple(img_points.reshape(-1, 2)[0])
        cv2.putText(
            display,
            label,
            (int(text_org[0]), int(text_org[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return display


class YoloUdpPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_udp_pose_node")

        self.declare_parameter("host_ip", "192.168.0.52")
        self.declare_parameter("port", 9999)
        self.declare_parameter("buff_size", 65536)
        self.declare_parameter("start_message", "Hello")
        self.declare_parameter(
            "model_path", "/home/addinedu/yolo_ws/src/inference/bolts-obb.pt"
        )
        self.declare_parameter(
            "calib_path",
            "/home/addinedu/yolo_ws/src/inference/"
            "camera_calib_upgraded_2026-01-12_16-17-24.npz",
        )
        self.declare_parameter("model_task", "obb")
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("object_width_m", 0.04)
        self.declare_parameter("object_height_m", 0.06)
        self.declare_parameter("topic_name", POSE_TOPIC)
        self.declare_parameter("timer_period_s", 0.01)
        self.declare_parameter("enable_display", True)
        self.declare_parameter("axis_length_m", 0.02)
        self.declare_parameter("filter_buffer_size", 5)

        self.host_ip = str(self.get_parameter("host_ip").value)
        self.port = int(self.get_parameter("port").value)
        self.buff_size = int(self.get_parameter("buff_size").value)
        self.start_message = str(self.get_parameter("start_message").value).encode()
        self.model_path = Path(str(self.get_parameter("model_path").value))
        self.calib_path = Path(str(self.get_parameter("calib_path").value))
        self.model_task = str(self.get_parameter("model_task").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.object_width_m = float(self.get_parameter("object_width_m").value)
        self.object_height_m = float(self.get_parameter("object_height_m").value)
        self.topic_name = str(self.get_parameter("topic_name").value)
        self.timer_period_s = float(self.get_parameter("timer_period_s").value)
        self.enable_display = bool(self.get_parameter("enable_display").value)
        self.axis_length_m = float(self.get_parameter("axis_length_m").value)
        self.filter_buffer_size = int(self.get_parameter("filter_buffer_size").value)
        if self.filter_buffer_size < 1:
            self.filter_buffer_size = 1
            self.get_logger().warning(
                "filter_buffer_size must be >= 1. Forced to 1."
            )
        self.raw_display_window_name = RAW_DISPLAY_WINDOW_NAME
        self.filtered_display_window_name = FILTERED_DISPLAY_WINDOW_NAME
        self.display_failed = False
        self.display_windows_initialized = False
        self.pose_history: dict[
            str, deque[tuple[np.ndarray, np.ndarray]]
        ] = {}

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.camera_matrix, self.dist_coeffs = load_intrinsics(self.calib_path)
        self.model = YOLO(str(self.model_path))

        self.pose_pub = self.create_publisher(YoloPose, self.topic_name, 10)

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, self.buff_size
        )
        self.client_socket.settimeout(0.001)
        self.client_socket.sendto(self.start_message, (self.host_ip, self.port))

        self.frame_count = 0
        self.timer = self.create_timer(self.timer_period_s, self._on_timer)

        self.get_logger().info(
            f"YOLO UDP Pose Node started -> {self.host_ip}:{self.port}, "
            f"topic: {self.topic_name}, filter_buffer_size={self.filter_buffer_size}"
        )

    def _filter_poses(
        self, poses: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        filtered_poses: list[dict[str, Any]] = []

        for pose in poses:
            pose_key = f'{pose["label"]}:{int(pose["det_idx"])}'
            tx, ty, tz = pose["xyz"]
            quat = np.array(rvec_to_quaternion(pose["rvec"]), dtype=np.float64)
            translation = np.array([tx, ty, tz], dtype=np.float64)

            if pose_key not in self.pose_history:
                self.pose_history[pose_key] = deque(maxlen=self.filter_buffer_size)

            self.pose_history[pose_key].append((translation, quat))

            history = self.pose_history[pose_key]
            translation_stack = np.stack([item[0] for item in history], axis=0)
            median_translation = np.median(translation_stack, axis=0)
            avg_quat = average_quaternions([item[1] for item in history])

            filtered_poses.append(
                {
                    **pose,
                    "rvec": quaternion_to_rvec(avg_quat),
                    "tvec": median_translation.reshape(3, 1),
                    "xyz": (
                        float(median_translation[0]),
                        float(median_translation[1]),
                        float(median_translation[2]),
                    ),
                    "quat": (
                        float(avg_quat[0]),
                        float(avg_quat[1]),
                        float(avg_quat[2]),
                        float(avg_quat[3]),
                    ),
                }
            )

        return filtered_poses

    def _ensure_display_windows(self, frame: np.ndarray) -> None:
        if self.display_windows_initialized:
            return

        height, width = frame.shape[:2]
        cv2.namedWindow(self.raw_display_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.filtered_display_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.raw_display_window_name, width, height)
        cv2.resizeWindow(self.filtered_display_window_name, width, height)
        self.display_windows_initialized = True

    def _on_timer(self) -> None:
        try:
            packet, _ = self.client_socket.recvfrom(self.buff_size)
        except socket.timeout:
            return
        except OSError as exc:
            self.get_logger().error(f"UDP recv error: {exc}")
            return

        try:
            data = base64.b64decode(packet, " /")
        except Exception:
            return

        npdata = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
        if frame is None:
            return

        result = self.model.predict(
            source=frame,
            task=self.model_task,
            conf=self.conf_thres,
            verbose=False,
        )[0]

        detections = extract_detection_candidates(result, self.conf_thres)
        poses = estimate_poses(
            detections,
            self.camera_matrix,
            self.dist_coeffs,
            self.object_width_m,
            self.object_height_m,
        )
        filtered_poses = self._filter_poses(poses)

        for pose in filtered_poses:
            msg = YoloPose()
            tx, ty, tz = pose["xyz"]
            qx, qy, qz, qw = pose["quat"]

            msg.part_class = str(pose["label"])
            msg.instance_id = int(pose["det_idx"])

            msg.pose.position.x = tx
            msg.pose.position.y = ty
            msg.pose.position.z = tz
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw
            self.pose_pub.publish(msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"frame={self.frame_count}, detections={len(detections)}, "
                f"pose_published={len(filtered_poses)}"
            )

        if self.enable_display and not self.display_failed:
            try:
                self._ensure_display_windows(frame)
                raw_display = draw_pose_overlay(
                    frame=frame,
                    poses=poses,
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=self.dist_coeffs,
                    axis_length_m=self.axis_length_m,
                )
                filtered_display = draw_pose_overlay(
                    frame=frame,
                    poses=filtered_poses,
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=self.dist_coeffs,
                    axis_length_m=self.axis_length_m,
                )
                cv2.imshow(self.raw_display_window_name, raw_display)
                cv2.imshow(self.filtered_display_window_name, filtered_display)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.display_failed = True
                self.get_logger().warning(
                    f"Display disabled (cv2.imshow error): {exc}"
                )

    def destroy_node(self) -> bool:
        try:
            self.client_socket.close()
        except Exception:
            pass
        if self.enable_display and not self.display_failed:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloUdpPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
