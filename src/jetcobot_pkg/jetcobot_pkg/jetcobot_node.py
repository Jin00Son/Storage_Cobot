#!/usr/bin/env python3
import time
import rclpy as rp
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from std_msgs.msg import Bool
from jetcobot_interfaces.action import PickAndPlace
from pymycobot.mycobot280 import MyCobot280

from jetcobot_pkg.utils.cobot_utils import(
    coords_replace_z
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
SELF_OFFSET_X = -15.0


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

        self.get_logger().info("✅ Robot Action Server started (with is_in_position checking)")
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

            ok_flag = self.mc.is_moving()  # 1:true, 0:false, -1:error
            print(ok_flag)

            if ok_flag == 0:
                return True, "REACHED"

            if ok_flag == -1:
                return False, "ERROR"

            if time.time() - t0 > timeout_sec:
                return False, "TIMEOUT"

            time.sleep(POLL_DT)

        return False, "ROS_NOT_OK"

    # -------------------------
    # Execute action (PickandPlace 동작 수행)
    # -------------------------
    def execute_cb(self, goal_handle):
        self.get_logger().info("🚀 Executing PickAndPlace action (state checked)")

        pick = list(goal_handle.request.pick_coords)     # [mm,mm,mm,deg,deg,deg]
        pick[0] += SELF_OFFSET_X
        place = list(goal_handle.request.place_coords)   # [mm,mm,mm,deg,deg,deg]

        result = PickAndPlace.Result()
        result.success = False
        result.message = ""

        # 0) Servo ON
        try:
            self.send_feedback(goal_handle, 0.02, "Servo ON")
            self.mc.focus_all_servos()
        except Exception as e:
            result.message = f"Servo ON failed: {e}"
            goal_handle.abort()
            return result

        # 1) Gripper OPEN
        self.send_feedback(goal_handle, 0.05, "Gripper OPEN")
        try:
            self.mc.set_gripper_value(GRIP_OPEN_VAL, GRIP_SPEED)
            time.sleep(0.3)
        except Exception as e:
            result.message = f"Gripper open failed: {e}"
            goal_handle.abort()
            return result

        # 2) PRE-PICK (안전 높이)
        pre_pick = coords_replace_z(pick, LIFT_MM)
        self.send_feedback(goal_handle, 0.12, "Move PRE-PICK")
        self.mc.send_coords(pre_pick, MOVE_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, pre_pick)
        if not ok:
            result.message = f"Move PRE-PICK failed: {why}"
            goal_handle.abort()
            return result

        # 3) PICK (내려가기)
        pick_cmd = coords_replace_z(pick, PICK_Z_MM)
        self.send_feedback(goal_handle, 0.28, "Move PICK")
        self.mc.send_coords(pick_cmd, PICK_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, pick_cmd)
        if not ok:
            result.message = f"Move PICK failed: {why}"
            goal_handle.abort()
            return result

        # 4) Gripper CLOSE
        self.send_feedback(goal_handle, 0.40, "Gripper CLOSE")
        try:
            self.mc.set_gripper_value(GRIP_CLOSE_VAL, GRIP_SPEED)
            time.sleep(0.4)
        except Exception as e:
            result.message = f"Gripper close failed: {e}"
            goal_handle.abort()
            return result

        # 5) Lift (PRE-PICK로 다시)
        self.send_feedback(goal_handle, 0.52, "Lift")
        self.mc.send_coords(pre_pick, PICK_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, pre_pick)
        if not ok:
            result.message = f"Lift failed: {why}"
            goal_handle.abort()
            return result

        # 6) PRE-PLACE
        pre_place = coords_replace_z(place, LIFT_MM)
        self.send_feedback(goal_handle, 0.65, "Move PRE-PLACE")
        self.mc.send_coords(pre_place, MOVE_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, pre_place)
        if not ok:
            result.message = f"Move PRE-PLACE failed: {why}"
            goal_handle.abort()
            return result

        # 7) PLACE (내려가기)
        self.send_feedback(goal_handle, 0.80, "Move PLACE")
        self.mc.send_coords(place, PLACE_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, place)
        if not ok:
            result.message = f"Move PLACE failed: {why}"
            goal_handle.abort()
            return result

        # 8) Gripper OPEN (release)
        self.send_feedback(goal_handle, 0.90, "Gripper OPEN (release)")
        try:
            self.mc.set_gripper_value(GRIP_OPEN_VAL, GRIP_SPEED)
            time.sleep(0.4)
        except Exception as e:
            result.message = f"Gripper open(release) failed: {e}"
            goal_handle.abort()
            return result

        # 9) HOME
        self.send_feedback(goal_handle, 0.95, "Return HOME")
        self.mc.send_coords(HOME_COORDS, MOVE_SPEED, MOVE_MODE)
        ok, why = self.wait_until_reached(goal_handle, HOME_COORDS, timeout_sec=15.0)
        if not ok:
            self.get_logger().warn(f"HOME move not confirmed: {why}")

        # DONE
        self.send_feedback(goal_handle, 1.00, "Done ✅")
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
