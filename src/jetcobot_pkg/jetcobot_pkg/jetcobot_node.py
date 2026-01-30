#!/usr/bin/env python3
import time
import rclpy as rp
import math
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from std_msgs.msg import Bool
from jetcobot_interfaces.action import PickAndPlace
from pymycobot.mycobot280 import MyCobot280

from jetcobot_pkg.utils.cobot_utils import(
    coords_replace_z,
    rotmat_to_euler_intrinsic_ZYX_deg,
    gripper_goal_to_ee_cmd_coords_mm_deg
)

# =========================
# ✅ 전역 설정
# =========================
MYCOBOT_PORT = "/dev/ttyJETCOBOT"
MYCOBOT_BAUD = 1000000

ACTION_NAME = "pickandplace"

MOVE_SPEED = 60        
MOVE_MODE = 0           
PICK_SPEED = 20
PLACE_SPEED = 30

# 그리퍼 값
GRIP_CLOSE_VAL = 20
GRIP_OPEN_VAL = 100
GRIP_SPEED = 50

# pick 동작 시 살짝 위로 뜨는 lift 높이(mm)
LIFT_MM = 150.0

# 도달 체크 폴링
POLL_DT = 0.05           # 50ms
MOVE_TIMEOUT_SEC = 15.0  # 각 모션 최대 허용 시간(원하면 늘려도 됨)

# 홈 포즈(원하면 수정)
HOME_COORDS = [-64.2, 23.2, 235.1, -150.48, 27.49, 142.74]  # [mm, mm, mm, deg, deg, deg]
HOME_ANGLES = [-90, 90, -90, -50, 0, 45]

# 실제 내려가는 end effector Z
PICK_Z_MM = 110.0
PLACE_Z_MM = 120.0

# 실제 오차값 
SELF_OFFSET_X = -10.0

# 로봇 위치 판단 허용 오차값
POS_FLAG_THRESH = 30.0

# end effector 기준 그리퍼 offset
GRIPPER_Z_OFFSET_DEG = -45.0
GRIPPER_Y_OFFSET_MM = -10.0
GRIPPER_Z_OFFSET_MM = 100.0


# =========================
# ✅ Jetcobot Node
# =========================
class JetcobotRobotActionServer(Node):
    def __init__(self):
        super().__init__("jetcobot_robot_action_server")

        self.mc = MyCobot280(MYCOBOT_PORT, MYCOBOT_BAUD)

        # 시작 시 홈
        try:
            self.mc.focus_all_servos()
            self.mc.send_angles(HOME_ANGLES, MOVE_SPEED, MOVE_MODE)
            time.sleep(2.0)
        except Exception as e:
            self.get_logger().error(f"Init HOME failed: {e}")

        # 완료 publish(선택)
        self.done_pub = self.create_publisher(Bool, "/pickandplace_done", 10)

        # =================
        # 📡 ROS 통신 
        # =================  
        self._server = ActionServer(
            self,
            PickAndPlace,
            ACTION_NAME,
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
        )

        self.get_logger().info("✅ Robot Action Server started (with is_moving checking)")
        self.get_logger().info(f"- action name: {ACTION_NAME}")
        self.get_logger().info(f"- port: {MYCOBOT_PORT} baud:{MYCOBOT_BAUD}")
    
    # =================
    # 🖨️ Node 함수
    # =================
    # -------------------------
    # Goal accept/reject
    # -------------------------
    def goal_cb(self, goal_request: PickAndPlace.Goal):
        if len(goal_request.pick_coords) != 6 or len(goal_request.place_coords) != 6:
            self.get_logger().error("❌ Goal rejected: pick/place coords must be length 6")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        self.get_logger().warn("⚠️ Cancel requested")
        return CancelResponse.ACCEPT

    # -------------------------
    # Feedback 함수
    # -------------------------
    def send_feedback(self, goal_handle, progress: float, state: str):
        fb = PickAndPlace.Feedback()
        fb.progress = float(progress)
        fb.state = str(state)
        goal_handle.publish_feedback(fb)

    # -------------------------
    # ⭐⭐ Wait until in position ⭐⭐ (상태판단 함수)
    # -------------------------
    def wait_until_reached(self, goal_handle, target_coords, timeout_sec=MOVE_TIMEOUT_SEC):
        """
        target_coords: [x,y,z,rx,ry,rz] (mm,deg)
        is_in_position(data, flag)
          flag=1 -> coords
        return: (ok:bool, reason:str)
        """
        t0 = time.time()
        while rp.ok():
            if goal_handle.is_cancel_requested:
                return False, "CANCEL"

            done_flag = self.mc.is_moving()  # 1:true, 0:false, -1:error    
            current_coords = self.mc.get_coords()
            pos_flag = 1
            for i in range(0,3):
                if abs(target_coords[i] - current_coords[i]) > POS_FLAG_THRESH: # 로봇 translation 위치만 확인 (회전X -> 특이점 발생)
                    pos_flag = 0
                    # print("value is over Thresh index:",i,", value", target_coords[i] - current_coords[i])

            if done_flag == 0 and pos_flag == 1:
                return True, "REACHED"

            if done_flag == -1:
                return False, "ERROR"

            if time.time() - t0 > timeout_sec:
                return False, "TIMEOUT. Robot Cannot Find IK Solution. Returning to Home Position"

            time.sleep(POLL_DT)

        return False, "ROS_NOT_OK"

    def sendcoords_flip_z(self, coords_mm_deg) -> np.ndarray:
        """
        입력: send_coords 형식 [x,y,z,rx,ry,rz] (mm, deg)
        - rx,ry,rz 는 intrinsic ZYX euler (deg) 로 해석
        출력: 3x3 Rotation matrix (np.float64)

        정의:
        R = Rz(rz) * Ry(ry) * Rx(rx)
        """
        if coords_mm_deg is None or len(coords_mm_deg) != 6:
            raise ValueError("coords_mm_deg must be length 6: [x,y,z,rx,ry,rz]")

        rx = float(coords_mm_deg[3]) * math.pi / 180.0
        ry = float(coords_mm_deg[4]) * math.pi / 180.0
        rz = float(coords_mm_deg[5]) * math.pi / 180.0


        _Rx = np.array([[1.0, 0.0, 0.0],
                     [0.0,  math.cos(rx), -math.sin(rx)],
                     [0.0,  math.sin(rx),  math.cos(rx)]], dtype=np.float64)
        
        _Ry = np.array([[ math.cos(ry), 0.0, math.sin(ry)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(ry), 0.0, math.cos(ry)]], dtype=np.float64)
        
        _Rz = np.array([[math.cos(rz), -math.sin(rz), 0.0],
                     [math.sin(rz),  math.cos(rz), 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

        R = _Rz @ _Ry @ _Rx

        R_flip_z_180 = np.array([
        [1.0,  0.0,  0.0],
        [0.0, 1.0,  0.0],
        [0.0,  0.0, -1.0],
        ], dtype=np.float64)

        Rm_cmd = R @ R_flip_z_180

        rx, ry, rz = rotmat_to_euler_intrinsic_ZYX_deg(Rm_cmd)

        return [float(coords_mm_deg[0]), float(coords_mm_deg[1]), float(coords_mm_deg[2]),
                    float(rx), float(ry), float(rz)]



    # -------------------------
    # Execute action (PickandPlace 동작 수행)
    # -------------------------
    def execute_cb(self, goal_handle):
        self.get_logger().info("🚀 Executing PickAndPlace action (state checked)")

        pick = self.sendcoords_flip_z(list(goal_handle.request.pick_coords))     # [mm,mm,mm,deg,deg,deg]
        place = list(goal_handle.request.place_coords)   # [mm,mm,mm,deg,deg,deg]

        # TCP 적용 (gripper 기준 목표 -> EE send_coords)
        pick_coords = gripper_goal_to_ee_cmd_coords_mm_deg(
            pick,
            gripper_z_offset_deg=GRIPPER_Z_OFFSET_DEG,
            gripper_y_offset_mm=GRIPPER_Y_OFFSET_MM,
            gripper_z_offset_mm=GRIPPER_Z_OFFSET_MM,
        )
        place_coords = gripper_goal_to_ee_cmd_coords_mm_deg(
            place,
            gripper_z_offset_deg=GRIPPER_Z_OFFSET_DEG,
            gripper_y_offset_mm=GRIPPER_Y_OFFSET_MM,
            gripper_z_offset_mm=GRIPPER_Z_OFFSET_MM,
        )

        pick_coords[0] += SELF_OFFSET_X


        result = PickAndPlace.Result()
        result.success = False
        result.message = ""

        # 0) Servo ON
        try:
            self.send_feedback(goal_handle, 10, "Servo ON")
            self.mc.focus_all_servos()
        except Exception as e:
            result.message = f"Servo ON failed: {e}"
            goal_handle.abort()
            return result

        # 1) Gripper OPEN
        self.send_feedback(goal_handle, 20, "Gripper OPEN")
        try:
            self.mc.set_gripper_value(GRIP_OPEN_VAL, GRIP_SPEED)
            time.sleep(0.3)
        except Exception as e:
            result.message = f"Gripper open failed: {e}"
            goal_handle.abort()
            return result

        # 2) PRE-PICK (안전 높이)
        pre_pick = coords_replace_z(pick_coords, LIFT_MM)
        self.send_feedback(goal_handle, 30, "Move PRE-PICK")
        self.mc.send_coords(pre_pick, MOVE_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, pre_pick)
        if not ok:
            result.message = f"Move PRE-PICK failed: {why}"
            goal_handle.abort()
            return result

        # 3) PICK (내려가기)
        pick_cmd = coords_replace_z(pick_coords, PICK_Z_MM)
        self.send_feedback(goal_handle, 40, "Move PICK")
        self.mc.send_coords(pick_cmd, PICK_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, pick_cmd)
        if not ok:
            result.message = f"Move PICK failed: {why}"
            goal_handle.abort()
            return result

        # 4) Gripper CLOSE
        self.send_feedback(goal_handle, 50, "Gripper CLOSE")
        try:
            self.mc.set_gripper_value(GRIP_CLOSE_VAL, GRIP_SPEED)
            time.sleep(0.4)
        except Exception as e:
            result.message = f"Gripper close failed: {e}"
            goal_handle.abort()
            return result

        # 5) Lift (PRE-PICK로 다시)
        self.send_feedback(goal_handle, 60, "Lift")
        self.mc.send_coords(pre_pick, PICK_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, pre_pick)
        if not ok:
            result.message = f"Lift failed: {why}"
            goal_handle.abort()
            return result

        # 6) PRE-PLACE
        pre_place = coords_replace_z(place_coords, LIFT_MM)
        self.send_feedback(goal_handle, 70, "Move PRE-PLACE")
        self.mc.send_coords(pre_place, MOVE_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, pre_place)
        if not ok:
            result.message = f"Move PRE-PLACE failed: {why}"
            goal_handle.abort()
            return result

        # 7) PLACE (내려가기)
        self.send_feedback(goal_handle, 80, "Move PLACE")
        self.mc.send_coords(place_coords, PLACE_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, place_coords)
        if not ok:
            result.message = f"Move PLACE failed: {why}"
            goal_handle.abort()
            return result

        # 8) Gripper OPEN (release)
        self.send_feedback(goal_handle, 90, "Gripper OPEN (release)")
        try:
            self.mc.set_gripper_value(GRIP_OPEN_VAL, GRIP_SPEED)
            time.sleep(0.4)
        except Exception as e:
            result.message = f"Gripper open(release) failed: {e}"
            goal_handle.abort()
            return result

        # 9) HOME
        self.send_feedback(goal_handle, 95, "Return HOME")
        self.mc.send_angles(HOME_ANGLES, MOVE_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, HOME_COORDS, timeout_sec=15.0)
        if not ok:
            self.get_logger().warn(f"HOME move not confirmed: {why}")
        

        # DONE
        self.send_feedback(goal_handle, 100, "Done ✅")
        result.success = True
        result.message = "Pick and Place done (checked)"

        goal_handle.succeed()

        done_msg = Bool()
        done_msg.data = True
        self.done_pub.publish(done_msg)

        return result


def main(args=None):
    rp.init(args=args)
    node = JetcobotRobotActionServer()
    try:
        rp.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rp.shutdown()


if __name__ == "__main__":
    main()
