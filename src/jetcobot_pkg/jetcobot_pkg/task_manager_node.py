#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node

from jetcobot_interfaces.msg import PartArray
from jetcobot_interfaces.action import PickAndPlace

from rclpy.action import ActionClient

from jetcobot_interfaces.srv import SetTaskMode, ManualPick

from jetcobot_pkg.utils.cobot_utils import(
    pose_mm_to_xyz_quat, # pose msg --> translation / rotation array로 분리하기
    quat_normalize, # 쿼터니언 normalize
    quat_mean_xyzw, # 쿼터니언 샘플들 평균 구하기
    quat_to_rotmat, # 쿼터니언 회전을 3x3 회전행렬로 변환하기
    rotmat_to_euler_intrinsic_ZYX_deg, # 3x3 회전행렬을 ZYX inrtinsic euler angle 회전으로 변환하기
    gripper_goal_to_ee_cmd_coords_mm_deg # ⭐⭐ TCP 적용 send_coords 보낼 명령 좌표 변환 ⭐⭐
) 

# =========================
# ✅ 전역 설정
# =========================
PARTS_TOPIC = "/parts" #구독 topic 이름
ACTION_NAME = "/pickandplace" #사용 action 이름

AUTO_STABLE_TIME_SEC = 3.0 # pick 할 부품 stable 시간 기준, 길수록 부품 선정 하는데 오래걸림, 너무 짧으면 오판 가능


SAMPLE_N = 30 # 측정 sample 프레임 개수, 증가할수록 측정 시간 오름


#===========📣 부품 창고 공간 확정되면 놓을곳 좌표 수정! 📣===========#

PLACE_COORDS_LIST = [
    [+80.0, 180.0, 10.0, 180.0, 0.0, 0.0], #place for id 1
    [0.0, 180.0, 10.0, 180.0, 0.0, 0.0], #place for id 2
    [-80.0, 180.0, 10.0, 180.0, 0.0, 0.0] #place for id 3                     
    ]

#=============================================================#


TICK_HZ = 20.0 #timer frequency

# (TCP) END EFFECTOR 와 GRIPPER 사이의 관계
GRIPPER_Z_OFFSET_DEG = -45.0 
GRIPPER_Y_OFFSET_MM = -10.0
GRIPPER_Z_OFFSET_MM = 100.0


# =========================
# ✅ 유틸 함수
# =========================

# 다수 sample 측정 후 tranlsation: medain, rotation: average로 부품 위치 추정
def robust_estimate_coords_mm(pose_mm_list):
    """
    return: [x,y,z,rx,ry,rz] (mm, deg)
      - xyz median
      - quat mean -> base->target rotation
      - ⭐ target +Z와 반대(-Z) 방향으로 자세 뒤집기(Rx 180)
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

    # -------------------------------
    # ⭐ base->target 회전
    # -------------------------------
    Rm = quat_to_rotmat(q_avg)

    # ✅ [ADD] target 좌표의 +Z와 반대(-Z)로 향하도록 뒤집기
    # Rx(180) = diag(1, -1, -1)  -> x축 유지, y/z 반전 => z축 방향 반전 효과
    R_flip_x_180 = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ], dtype=np.float64)

    # base->cmd = base->target * (target frame에서 x축 180도 회전)
    Rm_cmd = Rm @ R_flip_x_180

    rx, ry, rz = rotmat_to_euler_intrinsic_ZYX_deg(Rm_cmd)

    return [float(xyz_med[0]), float(xyz_med[1]), float(xyz_med[2]),
            float(rx), float(ry), float(rz)]


# =========================
# ✅ Task Manager Node
# =========================
class TaskManagerNode(Node):
    def __init__(self):

        super().__init__("task_manager_node")

        # =================
        # ✖️ Class 변수 
        # ================= 
        """
        ~~상태 변수~~
        AUTO_MODE: (True = 자동 모드),(False = 수동 모드),
        STATE:  (IDLE = jetcobot이 task가 없을 때),
                (SAMPLING = jetcobot에게 보낼 좌표를 sample 중),
                (EXECUTING = jetcobot이 현재 동작을 수행 중)
        """

        self.auto_mode = True       # default 자동 모드(바꾸고 싶을 시 False로 변경)
        self.parts = {}             # 구독한 /parts 저장 dictionary

        self.state = "IDLE"         # default IDLE 모드 - 시작시 바로 측정부터
        self.selected_id = None     # 선정된 부품 id 저장 변수
        self.sample_buf = []        # sample 측정시 버퍼 저장 list


        # =================
        # 📡 ROS 통신 
        # =================  
        self.sub = self.create_subscription(PartArray, PARTS_TOPIC, self.cb_parts, 10)
        self.cli_action = ActionClient(self, PickAndPlace, ACTION_NAME)

        self.srv_mode = self.create_service(SetTaskMode, "/set_task_mode", self.cb_set_mode)
        self.srv_manual = self.create_service(ManualPick, "/manual_pick", self.cb_manual_pick)

        self.timer = self.create_timer(1.0 / TICK_HZ, self.tick)

        self.get_logger().info("✅ TaskManagerNode started")


    # =================
    # 🖨️ Node 함수
    # =================

    def cb_parts(self, msg: PartArray): # /parts 토픽 구독 콜백함수
        """
        ☑️ IDLE -> self.parts에 구독한 정보 갱신
        ⭐ SAMPLING -> self.sample_buf에 특정 sample 만큼 저장
        """
        for part in msg.parts:
            self.parts[int(part.id)] = {
                "pose_mm": part.pose_mm,
                "ready": bool(part.ready_to_pick),
                "stable": float(part.stable_time_sec),
                "confidence": float(part.confidence),
                "last_seen": part.last_seen,
            }

        if self.state == "SAMPLING" and self.selected_id is not None:
            if self.selected_id in self.parts:
                p = self.parts[self.selected_id]["pose_mm"]
                self.sample_buf.append(p)
                if len(self.sample_buf) > SAMPLE_N:
                    self.sample_buf = self.sample_buf[-SAMPLE_N:]


    # =================
    # 📲 수동 모드 함수들
    # =================

    def cb_set_mode(self, req: SetTaskMode.Request, res: SetTaskMode.Response): # /set_task_mode 수동 모드 설정 서비스 콜백 함수
        self.auto_mode = bool(req.auto_mode)
        res.success = True
        res.message = f"auto_mode set to {self.auto_mode}"
        return res

    def cb_manual_pick(self, req: ManualPick.Request, res: ManualPick.Response): # /manual_pick 수동 모드 사용 서비스 콜백 함수
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


    # =================
    # ♻️ 자동 모드 함수
    # =================

    def tick(self): # timer 자동 루프 (자동 모드일시 작동)

        if self.state == "EXECUTING":  # EXECUTING --> 타이머 콜백 실행X
            return

        if self.state == "SAMPLING":   # SAMPLING --> 좌표 뽑아서 jetcobot_node 한테 action goal 보내기
            if self.selected_id is None:
                self.state = "IDLE"
                return

            if len(self.sample_buf) < SAMPLE_N:
                return
            
            # =================
            #  pick 좌표 선정
            # =================
            pick_coords = robust_estimate_coords_mm(self.sample_buf[:SAMPLE_N])
            if pick_coords is None:
                self._reset_to_idle()
                return
            
            # =================
            #  place 좌표 선정
            # =================
            if(self.selected_id == 1):
                place_coords = list(PLACE_COORDS_LIST[0])
            
            if(self.selected_id == 2):
                place_coords = list(PLACE_COORDS_LIST[1])
            
            if(self.selected_id == 3):
                place_coords = list(PLACE_COORDS_LIST[2])

            
            # =================
            #  그리퍼 기준 명령 좌표 변환
            # =================
            pick_coords = gripper_goal_to_ee_cmd_coords_mm_deg(
                pick_coords,
                gripper_z_offset_deg=GRIPPER_Z_OFFSET_DEG,
                gripper_y_offset_mm=GRIPPER_Y_OFFSET_MM,
                gripper_z_offset_mm=GRIPPER_Z_OFFSET_MM,
            )
            place_coords = gripper_goal_to_ee_cmd_coords_mm_deg(
                place_coords,
                gripper_z_offset_deg=GRIPPER_Z_OFFSET_DEG,
                gripper_y_offset_mm=GRIPPER_Y_OFFSET_MM,
                gripper_z_offset_mm=GRIPPER_Z_OFFSET_MM,
            )

            self._send_pick_and_place_goal(pick_coords, place_coords)
            self.state = "EXECUTING"
            return

        if self.state == "IDLE": # IDLE --> pick할 후보(candidate) 선정 (선정 우선순위: 가장 먼저 stable time 도달 >> 부품 id 순서)
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

    # =================
    #  🎲 기타 유틸 함수들
    # =================

    def _send_pick_and_place_goal(self, pick_coords, place_coords): # pick, place coords 받아서 Action goal 보내는 유틸 함수
        if not self.cli_action.wait_for_server(timeout_sec=1.0):
            self._reset_to_idle()
            return

        goal = PickAndPlace.Goal()
        goal.pick_coords = [float(x) for x in pick_coords]
        goal.place_coords = [float(x) for x in place_coords]

        send_future = self.cli_action.send_goal_async(goal, feedback_callback=self._feedback_cb)
        send_future.add_done_callback(self._goal_response_cb)

    def _feedback_cb(self, fb_msg): # action feedback 콜백 함수
        fb = fb_msg.feedback
        self.get_logger().info(f"[FB] {fb.progress:.1f}% {fb.state}")

    def _goal_response_cb(self, future): # action goal response 콜백 함수
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._reset_to_idle()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future): # action goal result 콜백 함수
        _ = future.result().result
        self._reset_to_idle()

    def _reset_to_idle(self): # 상태 + 내부 db 초기화 함수
        self.state = "IDLE"
        self.selected_id = None
        self.sample_buf = []
        self.parts = {}


# =================
#  메인 루프
# =================

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
