#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from jetcobot_interfaces.msg import PartArray
from jetcobot_interfaces.srv import SetTaskMode, ManualPick

from jetcobot_pkg.utils.cobot_utils import (
    pose_mm_to_xyz_quat,
    quat_normalize,
    quat_mean_xyzw,
    quat_to_rotmat,
    rotmat_to_euler_intrinsic_ZYX_deg,
)

from jetcobot_pkg.utils.jetcobot_action_client import (
    PickClient,
    MoveToPoseClient,
    PlaceClient
)

# =========================
# ✅ 전역 설정
# =========================
PARTS_TOPIC = "/parts"
ASSEMBLY_TOPIC = "/assembly_start"
STORAGE_TOPIC = "/storage_start"
ACTION_NAME = "/pickandplace"

AUTO_STABLE_TIME_SEC = 2.0
SAMPLE_N = 20

WAITING_ANGLES = [90, 90, -90, -50, 0, 45]
HOME_ANGLES = [-90, 90, -90, -50, 0, 45]
SAFE_ANGLES = [0, 90, -90, -50, 0, 45]

BOX_HEIGHT = 24.5
BASE_HEIGHT = 2.5

PLACE_Z_MM = BOX_HEIGHT - BASE_HEIGHT

PLACE_COORDS_LIST = [
    [ 0.0, 180.0, PLACE_Z_MM, 180.0, 0.0, 180.0],  # id 1
    [ 0.0, 220.0, PLACE_Z_MM, 180.0, 0.0, 180.0],  # id 2
    [ 0.0, 260.0, PLACE_Z_MM, 180.0, 0.0, 180.0],  # id 3
]

SAFE_PLACE_COORDS = [ 200.0, 0.0, PLACE_Z_MM, 180.0, 0.0, 0.0]

TICK_HZ = 20.0

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
        self.msg = Bool()
        self.msg.data = False
        self.assembly_start = False

        # ✅ [ADD] action에 사용할 pick/place 목표 좌표 저장
        self.pick_coords = None
        self.place_coords = None

        #topics
        self.pub_start = self.create_publisher(Bool, STORAGE_TOPIC, 10)

        # subscribe
        self.sub_parts = self.create_subscription(PartArray, PARTS_TOPIC, self.cb_parts, 10)
        self.sub_assembly = self.create_subscription(Bool, ASSEMBLY_TOPIC, self.cb_cobotcomms, 10)

        # services
        self.srv_mode = self.create_service(SetTaskMode, "/set_task_mode", self.cb_set_mode)
        self.srv_manual = self.create_service(ManualPick, "/manual_pick", self.cb_manual_pick)
        
        # action clients
        self.pick_cli = PickClient(self, action_name="/pick")
        self.move_cli = MoveToPoseClient(self, action_name="/move_to_pose")
        self.place_cli = PlaceClient(self, action_name="/place")

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

    def cb_cobotcomms(self, msg: Bool):
        # if not self.state == 'EXECUTING_WAIT':
        #     return
        self.assembly_start = bool(msg.data)

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
        # ✅ 항상 publish는 유지
        self.pub_start.publish(self.msg)

        # ✅ IDLE
        if self.state == "IDLE":
            self.msg.data = False

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
            return

        # ✅ SAMPLING
        if self.state == "SAMPLING":
            self.msg.data = False

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
            if self.selected_id // 1000 == 1:
                place_coords = list(PLACE_COORDS_LIST[0])
            elif self.selected_id // 1000 == 2:
                place_coords = list(PLACE_COORDS_LIST[1])
            elif self.selected_id // 1000 == 3:
                place_coords = list(PLACE_COORDS_LIST[2])
            else:
                place_coords = list(PLACE_COORDS_LIST[1])

            self.pick_coords = pick_coords
            self.place_coords = place_coords
            self.safe_pick = True
            self.safe_place = True

            if not self.pick_cli.send_goal(self.pick_coords, self.safe_pick):
                self.get_logger().error("send_goal failed.. Trying Again")
                return

            self.state = "EXECUTING_PICK"
            return

        # ✅ EXECUTING_PICK
        if self.state == "EXECUTING_PICK":
            self.msg.data = False

            # pick action done 확인
            if not self._is_action_done(self.pick_cli.action_done()):
                return
            
            # move to pose action 보내기
            if not self.move_cli.send_goal_angles(WAITING_ANGLES):
                self.get_logger().error("send_goal failed.. Trying Again")
                return

            self.state = "EXECUTING_WAIT_POSE"
            return

        # ✅ EXECUTING_WAIT_POSE
        if self.state == "EXECUTING_WAIT_POSE":
            self.msg.data = False

            if not self._is_action_done(self.move_cli.action_done()):
                return

            if self.assembly_start == True:
                self.get_logger().info(f"[TASK DONE] Waiting until Assembly Cobot Leaves Storage Area..")
                self.state = "EXECUTING_WAITING"
                return

            if not self.place_cli.send_goal(self.place_coords, self.safe_place):
                self.get_logger().error("send_goal failed.. Trying Again")
                return

            self.msg.data = True
            self.state = "EXECUTING_PLACE"
            return

        # ✅ EXECUTING_WAITING
        if self.state == "EXECUTING_WAITING":
            self.msg.data = False

            if self.assembly_start == False:
                if not self.place_cli.send_goal(self.place_coords, self.safe_place):
                    self.get_logger().error("send_goal failed.. Trying Again")
                    return

                self.msg.data = True
                self.state = "EXECUTING_PLACE"
            return

        # ✅ EXECUTING_PLACE
        if self.state == "EXECUTING_PLACE":
            self.msg.data = True

            if not self._is_action_done(self.place_cli.action_done()):
                return

            if not self.move_cli.send_goal_angles(HOME_ANGLES):
                self.get_logger().error("send_goal failed.. Trying Again")
                return

            self.msg.data = False
            self.state = "EXECUTING_HOME_POSE"
            return

        # ✅ EXECUTING_HOME_POSE
        if self.state == "EXECUTING_HOME_POSE":
            self.msg.data = False

            if not self._is_action_done(self.move_cli.action_done()):
                return

            self._reset_to_idle()
            return
        
        # ✅ EXECUTING_SAFE_MOVE
        if self.state == "EXECUTING_SAFE_MOVE":
            self.msg.data = False

            if not self._is_action_done(self.move_cli.action_done()):
                self.get_logger().error("send_goal failed.. Trying Again")
                return
            
            self.safe_place = True
            if not self.place_cli.send_goal(SAFE_PLACE_COORDS, self.safe_place):
                self.get_logger().error("send_goal failed.. Trying Again")
                return

            self.msg.data = False
            self.state = "EXECUTING_PLACE"
            return
        
    def _is_action_done(self, done):

        if done is None:
            return False
        success, message = done
        if success:
            self.get_logger().info(f"[TASK DONE] success={success} msg={message}")
            return True
        else:
            self.get_logger().error(f"[TASK FAIL] success={success} msg={message}")
            if self.state == "EXECUTING_PICK":
                self.get_logger().error(f"[TASK FAIL] Object is out of Cobot's Range. Replace Object! <Returning to Scanning State> 💨")
            self._reset_to_idle()
            return False

    def _reset_to_idle(self):

        if self.state == "EXECUTING_WAIT_POSE": # wait pose 이동 실패시
            self.safe_place = True
            if not self.place_cli.send_goal(self.pick_coords, self.safe_place): 
                self.get_logger().error("🛑 Safety Measures failed.. Breaking Systems, 👷 Manual Assistance Needed") # 추후 시스템 정지, 경고 보내는 기능 여기에 추가
        
            self.state = "EXECUTING_PLACE"
            
        elif self.state == "EXECUTING_PLACE": # Place 실패시: Safety Pose 이동 -> 안전 장소에 물체 두기 -> Home Pose 복귀
            self.msg.data = False # action을 실패했기에 False로 변경
            if not self.move_cli.send_goal_angles(SAFE_ANGLES): # SAFE_ANGLES는 실패 안한다고 가정
                self.get_logger().error("🛑 Safety Measures failed.. Breaking System, 👷 Manual Assistance Needed") # 추후 시스템 정지, 경고 보내는 기능 여기에 추가
            self.state = "EXECUTING_SAFE_MOVE"

        else: 
            self.state = "IDLE"
                
        self.selected_id = None
        self.sample_buf = []
        self.parts = {}  # ✅ 누적 방지: DB 초기화
        self.pick_coords = None
        self.place_coords = None

        



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
