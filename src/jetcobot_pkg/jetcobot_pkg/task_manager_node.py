#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node

from jetcobot_interfaces.msg import PartArray
from jetcobot_interfaces.srv import SetTaskMode, ManualPick

from jetcobot_pkg.utils.cobot_utils import (
    pose_mm_to_xyz_quat,
    quat_normalize,
    quat_mean_xyzw,
    quat_to_rotmat,
    rotmat_to_euler_intrinsic_ZYX_deg,
)

# ✅ [ADD] ActionClient 유틸 import
from jetcobot_pkg.utils.pickandplace_client import PickAndPlaceClient


# =========================
# ✅ 전역 설정
# =========================
PARTS_TOPIC = "/parts"
ACTION_NAME = "/pickandplace"

AUTO_STABLE_TIME_SEC = 2.0
SAMPLE_N = 30

PLACE_COORDS_LIST = [
    [+80.0, 180.0, 10.0, 180.0, 0.0, 180.0],  # id 1
    [  0.0, 180.0, 10.0, 180.0, 0.0, 180.0],  # id 2
    [-80.0, 180.0, 10.0, 180.0, 0.0, 180.0],  # id 3
]

TICK_HZ = 20.0

GRIPPER_Z_OFFSET_DEG = -45.0
GRIPPER_Y_OFFSET_MM = -10.0
GRIPPER_Z_OFFSET_MM = 100.0


# =========================
# ✅ 유틸 함수
# =========================
def robust_estimate_coords_mm(pose_mm_list):
    """
    return: [x,y,z,rx,ry,rz] (mm, deg)
      - xyz median
      - quat mean -> base->target rotation
      - target +Z와 반대(-Z) 방향으로 자세 뒤집기(Rx 180)
    """
    if len(pose_mm_list) == 0:
        return None

    xyz_list = []
    q_list = []

    for p in pose_mm_list:
        xyz, q = pose_mm_to_xyz_quat(p)
        xyz_list.append(xyz)
        q_list.append(quat_normalize(q))

    xyz_stack = np.stack(xyz_list, axis=0)
    xyz_med = np.median(xyz_stack, axis=0)

    q_avg = quat_mean_xyzw(q_list)
    if q_avg is None:
        return None

    Rm = quat_to_rotmat(q_avg)

    rx, ry, rz = rotmat_to_euler_intrinsic_ZYX_deg(Rm)

    return [float(xyz_med[0]), float(xyz_med[1]), float(xyz_med[2]),
            float(rx), float(ry), float(rz)]


# =========================
# ✅ Task Manager Node
# =========================
class TaskManagerNode(Node):
    def __init__(self):
        super().__init__("task_manager_node")

        self.auto_mode = True
        self.parts = {}

        self.state = "IDLE"     # IDLE / SAMPLING / EXECUTING
        self.selected_id = None
        self.sample_buf = []

        # subscribe
        self.sub = self.create_subscription(PartArray, PARTS_TOPIC, self.cb_parts, 10)

        # services
        self.srv_mode = self.create_service(SetTaskMode, "/set_task_mode", self.cb_set_mode)
        self.srv_manual = self.create_service(ManualPick, "/manual_pick", self.cb_manual_pick)

        # ✅ [MOD] ActionClient는 유틸로 대체
        self.pp = PickAndPlaceClient(node=self, action_name=ACTION_NAME, wait_server_timeout_sec=1.0)

        self.timer = self.create_timer(1.0 / TICK_HZ, self.tick)

        self.get_logger().info("✅ TaskManagerNode started")

    def cb_parts(self, msg: PartArray):
        for part in msg.parts:
            self.parts[int(part.id)] = {
                "pose_mm": part.pose_mm,
                "ready": bool(part.ready_to_pick),
                "stable": float(part.stable_time_sec)
            }

        if self.state == "SAMPLING" and self.selected_id is not None:
            if self.selected_id in self.parts:
                p = self.parts[self.selected_id]["pose_mm"]
                self.sample_buf.append(p)
                if len(self.sample_buf) > SAMPLE_N:
                    self.sample_buf = self.sample_buf[-SAMPLE_N:]

    def cb_set_mode(self, req: SetTaskMode.Request, res: SetTaskMode.Response):
        self.auto_mode = bool(req.auto_mode)
        res.success = True
        res.message = f"auto_mode set to {self.auto_mode}"
        return res

    def cb_manual_pick(self, req: ManualPick.Request, res: ManualPick.Response):
        part_id = int(req.part_id)

        if self.auto_mode:
            res.accepted = False
            res.message = "ManualPick rejected: auto_mode=True"
            return res

        if self.state != "IDLE":
            res.accepted = False
            res.message = f"ManualPick rejected: busy (state={self.state})"
            return res

        if part_id not in self.parts:
            res.accepted = False
            res.message = f"ManualPick rejected: part_id={part_id} not seen"
            return res

        if not self.parts[part_id]["ready"]:
            res.accepted = False
            res.message = f"ManualPick rejected: part_id={part_id} ready_to_pick=False"
            return res

        self.selected_id = part_id
        self.sample_buf = []
        self.state = "SAMPLING"

        res.accepted = True
        res.message = f"ManualPick accepted: sampling part_id={part_id}"
        return res

    def tick(self):
        # ✅ [ADD] EXECUTING 중이면 action 완료를 consume_done()으로 감지
        if self.state == "EXECUTING":
            done = self.pp.action_done()
            if done is None:
                return

            success, message = done
            self.get_logger().info(f"[TASK DONE] success={success} msg={message}")
            self._reset_to_idle()
            return

        # ✅ SAMPLING
        if self.state == "SAMPLING":
            if self.selected_id is None:
                self.state = "IDLE"
                return

            if len(self.sample_buf) < SAMPLE_N:
                return

            pick_coords = robust_estimate_coords_mm(self.sample_buf[:SAMPLE_N])
            if pick_coords is None:
                self._reset_to_idle()
                return

            # place coords by id
            if self.selected_id == 1:
                place_coords = list(PLACE_COORDS_LIST[0])
            elif self.selected_id == 2:
                place_coords = list(PLACE_COORDS_LIST[1])
            elif self.selected_id == 3:
                place_coords = list(PLACE_COORDS_LIST[2])
            else:
                place_coords = list(PLACE_COORDS_LIST[1])

            # ✅ [MOD] goal 전송은 유틸로 한 줄
            ok = self.pp.send_goal(pick_coords, place_coords)
            if not ok:
                self.get_logger().error("send_goal failed")
                self._reset_to_idle()
                return

            self.state = "EXECUTING"
            return

        # ✅ IDLE
        if self.state == "IDLE":
            if not self.auto_mode:
                return

            candidates = [pid for pid, info in self.parts.items() if info["stable"] >= AUTO_STABLE_TIME_SEC]
            if not candidates:
                return

            candidates.sort()
            chosen = candidates[0]

            self.selected_id = chosen
            self.sample_buf = []
            self.state = "SAMPLING"

    def _reset_to_idle(self):
        self.state = "IDLE"
        self.selected_id = None
        self.sample_buf = []
        self.parts = {}  # ✅ 누적 방지: DB 초기화


def main():
    rclpy.init()
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
