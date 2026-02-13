import json  # ★ JSON 처리를 위해 추가
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Int32, Float32, Bool
from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped


# **손진우 추가 내용** interface 추가####################################33
from jetcobot_interfaces.msg import SectionResult, StorageRequest, StorageResponse, ManualRequest
#####################################################################33


import mysql.connector
from PyQt6.QtCore import QThread, pyqtSignal

# 1. 모바일 로봇 관련
T_MOBILE_PREFIX = "/pinky"

T_SUFFIX_POSE    = "/amcl_pose"    # 위치 정보 (PoseWithCovarianceStamped)
T_SUFFIX_BATTERY = "/battery/present" # 배터리 정보 (Float32)
T_SUFFIX_STATE   = "/state"   # 상태 정보 (String)
T_SUFFIX_CMD     = "/cmd"     # 이동 명령 (String)
T_SUFFIX_LOAD_DONE = "/load_done"   # 상차 완료 (Bool)
T_SUFFIX_UNLOAD_DONE = "/unload_done" # 하차 완료 (Bool)
T_SUFFIX_MOVE_ROLE = "/move_role"   # 이동 역할 (String)

# 네임스페이스 없이 로봇 1대만 테스트할 때 사용
SINGLE_ROBOT_MODE = False
SINGLE_ROBOT_ID = "pinky1"
SINGLE_POSE_TOPIC = "/amcl_pose"
SINGLE_BATTERY_TOPIC = "/battery/present"
SINGLE_STATE_TOPIC = "/state"
SINGLE_CMD_TOPIC = "/cmd"
SINGLE_LOAD_DONE_TOPIC = "/load_done"
SINGLE_UNLOAD_DONE_TOPIC = "/unload_done"
SINGLE_MOVE_ROLE_TOPIC = "/move_role"

# 2. 로봇팔 관련
T_ARM_UNLOAD_SIGNAL = "/warehouse/unload"     # [수신] 로봇팔이 출고 완료했을 때 (Int)
T_ARM_TARGET_SLOT   = "/robot_arm/target_slot" # [송신] 로봇팔에게 "여기다 넣어" 명령 (String)

#####################################################################################################################
# 3. Jetcobot 관련
T_JETCO_REQ = "/jetcobot/storage/auto/request"   # [수신] 요청
T_JETCO_RES = "/jetcobot/storage/auto/response"  # [발행] 응답
T_JETCO_STR_UPD = "/jetcobot/storage/db_update"              # [수신] DB 업데이트
T_JETCO_ASS_UPD = "/jetcobot/assembly/db_update"
T_JETCO_ASS_BOOT = "/jetcobot/assembly/boot"
T_JETCO1_STATUS = "/jetcobot/storage/status"
T_JETCO1_SETMODE = "/jetcobot/storage/set_mode"
T_JETCO1_MANUAL_REQ = "/jetcobot/storage/manual/request"
T_JETCO1_MANUAL_RES = "/jetcobot/storage/manual/response"
T_JETCO1_STR_BOOT = '/jetcobot/storage/boot'
T_JETCO2_STACK_REQ = "/jetcobot/assembly/stack/request"

#####################################################################################################################

# 4. DB 설정
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASS = '1234'
DB_NAME = 'smart_factory'

# 5. 지도 설정
REAL_MAP_WIDTH_CM = 203
REAL_MAP_HEIGHT_CM = 83
MAP_OFFSET_X_CM = REAL_MAP_WIDTH_CM / 2.0
MAP_OFFSET_Y_CM = REAL_MAP_HEIGHT_CM / 2.0

# ★ [핵심] GUI 노드 (ROS 2 통신 담당)
class GuiNode(QThread):
    robot_update_signal = pyqtSignal(dict)  
    unload_signal = pyqtSignal(int)
    jetco_log_signal = pyqtSignal(str) # Jetcobot 로그용 신호e

    ####################################################33
    jetco_storage_boot_signal = pyqtSignal(bool)
    jetco_storage_status_signal = pyqtSignal(str)
    ######################################################


    def __init__(self):
        super().__init__()
        self.node = None
        self.running = True
        self.cmd_pubs = {} 
        self.load_done_pubs = {}
        self.unload_done_pubs = {}
        self.move_role_pubs = {}
        self.arm_pub = None 
        self.jetco_res_pub = None # Response 발행용

    def run(self):
        rclpy.init()
        self.node = Node('smartfactory_gui_node')
        amcl_pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # 1. Pinky 로봇 설정
        if SINGLE_ROBOT_MODE:
            # (기존 SINGLE 모드 코드 생략 없이 유지)
            robot_name = SINGLE_ROBOT_ID
            self.node.create_subscription(PoseWithCovarianceStamped, SINGLE_POSE_TOPIC,
                                          lambda m, r=robot_name: self.pose_callback(m, r), amcl_pose_qos)
            self.node.create_subscription(Float32, SINGLE_BATTERY_TOPIC,
                                          lambda m, r=robot_name: self.battery_callback(m, r), 10)
            self.node.create_subscription(String, SINGLE_STATE_TOPIC,
                                          lambda m, r=robot_name: self.state_callback(m, r), 10)
            self.cmd_pubs[f"/{robot_name}"] = self.node.create_publisher(String, SINGLE_CMD_TOPIC, 10)
            self.load_done_pubs[f"/{robot_name}"] = self.node.create_publisher(Bool, SINGLE_LOAD_DONE_TOPIC, 10)
            self.unload_done_pubs[f"/{robot_name}"] = self.node.create_publisher(Bool, SINGLE_UNLOAD_DONE_TOPIC, 10)
            self.move_role_pubs[f"/{robot_name}"] = self.node.create_publisher(String, SINGLE_MOVE_ROLE_TOPIC, 10)
        else:
            for i in range(1, 4):
                robot_name = f"{T_MOBILE_PREFIX}{i}"
                self.node.create_subscription(PoseWithCovarianceStamped, f"{robot_name}{T_SUFFIX_POSE}",
                                              lambda m, r=robot_name: self.pose_callback(m, r), amcl_pose_qos)
                self.node.create_subscription(Float32, f"{robot_name}{T_SUFFIX_BATTERY}",
                                              lambda m, r=robot_name: self.battery_callback(m, r), 10)
                self.node.create_subscription(String, f"{robot_name}{T_SUFFIX_STATE}",
                                              lambda m, r=robot_name: self.state_callback(m, r), 10)
                self.cmd_pubs[robot_name] = self.node.create_publisher(String, f"{robot_name}{T_SUFFIX_CMD}", 10)
                self.load_done_pubs[robot_name] = self.node.create_publisher(Bool, f"{robot_name}{T_SUFFIX_LOAD_DONE}", 10)
                self.unload_done_pubs[robot_name] = self.node.create_publisher(Bool, f"{robot_name}{T_SUFFIX_UNLOAD_DONE}", 10)
                self.move_role_pubs[robot_name] = self.node.create_publisher(String, f"{robot_name}{T_SUFFIX_MOVE_ROLE}", 10)

        
        # **손진우 추가 내용** ros통신 수정
        # 2. 로봇팔 (Jetcobot) 설정 [★추가된 부분]
        # Request 구독
        self.node.create_subscription(StorageRequest, T_JETCO_REQ, self.callback_jetco_request, 10)
        # Status 구독
        self.node.create_subscription(String, T_JETCO1_STATUS, self.callback_status, 10)
        # Update 구독
        self.node.create_subscription(SectionResult, T_JETCO_STR_UPD, self.callback_jetco_update, 10)
        self.node.create_subscription(SectionResult, T_JETCO_ASS_UPD, self.callback_jetco_update, 10)
        # Response 발행
        self.jetco_res_pub = self.node.create_publisher(StorageResponse, T_JETCO_RES, 10)
        self.jetco_mode_pub = self.node.create_publisher(Int32, T_JETCO1_SETMODE, 10)
        # boot 발행
        self.jetco_storage_boot_pub = self.node.create_publisher(Bool, T_JETCO1_STR_BOOT,10)
        self.jetco_assembly_boot_pub = self.node.create_publisher(Bool, T_JETCO_ASS_BOOT,10)
        self.jetco_storage_manual_req_pub = self.node.create_publisher(ManualRequest,T_JETCO1_MANUAL_REQ,10)
        self.jetco_assembly_stack_req_pub = self.node.create_publisher(Int32,T_JETCO2_STACK_REQ, 10)
        # **손진우 추가 내용** ros통신 수정


        # Arm 통신
        self.node.create_subscription(Int32, T_ARM_UNLOAD_SIGNAL, self.unload_callback, 10)
        self.arm_pub = self.node.create_publisher(String, T_ARM_TARGET_SLOT, 10)

        # OpenManipulator Start Publisher Test1
        self.manip_start_pub = self.node.create_publisher(Bool,'/pick_and_place/start',10)

       

        print(f"GUI 노드 시작 (토픽 설정 완료)")
        while self.running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
        self.node.destroy_node()
        rclpy.shutdown()

############3###############################################################################################################################################3
    # -------------------------------------------------------------
    # ★ [기능 1] Request 처리: DB 확인 후 Response 발행
    def callback_jetco_request(self, msg):
        try:      
            target_section = str(msg.section)
            task = int(msg.task_id) # 0: pick, 1: place
            found_id = 0 # 0이면 실패 또는 없음

            self.jetco_log_signal.emit(f"📩 [요청] 구역:{target_section}, 작업:{'입고' if task==1 else '출고'}({task})")

            conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
            cursor = conn.cursor()

            # if task == 0: # Pick (출고): 물건이 있는(1) 자리 찾기
            #     sql = "SELECT current_part_id FROM warehouse_slots WHERE section=%s AND is_occupied=1 LIMIT 1"
            #     cursor.execute(sql, (target_section,))
            #     res = cursor.fetchone()
            #     if res and res[0]: 
            #         found_id = res[0] # 꺼낼 물건 ID

            if task == 1: # Place (입고): 빈(0) 자리 찾기
                sql = "SELECT slot_id FROM warehouse_slots WHERE section=%s AND is_occupied=0"
                cursor.execute(sql, (target_section,))
                cnt = cursor.fetchone()[0]
                # 빈 자리가 있으면 성공 신호로 999 (또는 실제 넣을 ID) 리턴
            conn.close()

            # Response 전송
            result_msg = StorageResponse()
            result_msg.section = target_section
            if cnt is None:
                result_msg.id = None
            else:
                result_msg.id = int(cnt[2])
            res_data = {"section": target_section, "id": str(cnt[2])}
            self.jetco_res_pub.publish(result_msg)
            self.jetco_log_signal.emit(f"[응답] {res_data}")

        except Exception as e:
            print(f"Jetcobot Request Error: {e}")

    # -------------------------------------------------------------
    # ★ [기능 2] Update 처리: DB 상태 실제 변경
    def callback_jetco_update(self, msg):
        try:

            sec = str(msg.section)
            pid = int(msg.id)
            occ = int(msg.occupy) # 0: 비움, 1: 채움

            slot_id = sec+"-"+str(pid) # A-1 꼴로 변경
            
            action_str = "채움" if occ == 1 else "비움"
            self.jetco_log_signal.emit(f"🔄 [DB수정] {sec}구역, ID:{pid} -> {action_str}")

            conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
            cursor = conn.cursor()

            if occ == 1: # 채우기 (Place 완료)
                # 빈 슬롯 하나를 해당 ID로 채움
                sql = """UPDATE warehouse_slots SET is_occupied=1
                         WHERE slot_id=%s AND is_occupied=0 LIMIT 1"""
                cursor.execute(sql, (slot_id,))
            else: # 비우기 (Pick 완료)
                # 빈 슬롯 하나를 해당 ID로 비움
                sql = """UPDATE warehouse_slots SET is_occupied=0
                         WHERE slot_id=%s AND is_occupied=1 LIMIT 1"""
                cursor.execute(sql, (slot_id,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Jetcobot Update Error: {e}")

    def callback_status(self, msg):
        self.jetco_storage_status_signal.emit(str(msg.data))



        



############3###############################################################################################################################################3


    def pose_callback(self, msg, robot_id):
        clean_id = robot_id.replace("/", "")
        x_m = msg.pose.pose.position.x
        y_m = msg.pose.pose.position.y
        x_cm = int(x_m * 100.0 + MAP_OFFSET_X_CM)
        y_cm = int(y_m * 100.0 + MAP_OFFSET_Y_CM)
        data = {"id": clean_id, "location": f"{x_cm},{y_cm}"}
        self.robot_update_signal.emit(data)

    def battery_callback(self, msg, robot_id):
        clean_id = robot_id.replace("/", "")
        data = {"id": clean_id, "battery": msg.data}
        self.robot_update_signal.emit(data)

    def state_callback(self, msg, robot_id):
        clean_id = robot_id.replace("/", "")
        data = {"id": clean_id, "state": msg.data}
        self.robot_update_signal.emit(data)

    def unload_callback(self, msg):
        self.unload_signal.emit(msg.data)

    def send_command(self, robot_id, cmd_str):
        target_key = robot_id if robot_id.startswith("/") else f"/{robot_id}"
        if target_key not in self.cmd_pubs:
            for key in self.cmd_pubs.keys():
                if robot_id in key:
                    target_key = key
                    break
        if target_key in self.cmd_pubs:
            msg = String()
            msg.data = cmd_str
            self.cmd_pubs[target_key].publish(msg)
            print(f"명령 전송 [{target_key}]: {cmd_str}")
        else:
            print(f"❌ 로봇을 찾을 수 없음: {robot_id}")

    def send_load_done(self, robot_id):
        target_key = robot_id if robot_id.startswith("/") else f"/{robot_id}"
        if target_key not in self.load_done_pubs:
            for key in self.load_done_pubs.keys():
                if robot_id in key:
                    target_key = key; break
        if target_key in self.load_done_pubs:
            msg = Bool(); msg.data = True
            self.load_done_pubs[target_key].publish(msg)
            print(f"상차 완료 전송 [{target_key}]: True")
        else: print(f"❌ 로봇을 찾을 수 없음: {robot_id}")

    def send_unload_done(self, robot_id):
        target_key = robot_id if robot_id.startswith("/") else f"/{robot_id}"
        if target_key not in self.unload_done_pubs:
            for key in self.unload_done_pubs.keys():
                if robot_id in key:
                    target_key = key; break
        if target_key in self.unload_done_pubs:
            msg = Bool(); msg.data = True
            self.unload_done_pubs[target_key].publish(msg)
            print(f"하차 완료 전송 [{target_key}]: True")
        else: print(f"❌ 로봇을 찾을 수 없음: {robot_id}")
        
    def send_move_role(self, robot_id, role_id):
        target_key = robot_id if robot_id.startswith("/") else f"/{robot_id}"
        if target_key not in self.move_role_pubs:
            for key in self.move_role_pubs.keys():
                if robot_id in key:
                    target_key = key; break
        if target_key in self.move_role_pubs:
            msg = String(); msg.data = str(role_id)
            self.move_role_pubs[target_key].publish(msg)
            print(f"이동 역할 전송 [{target_key}]: {msg.data}")
        else: print(f"❌ 로봇을 찾을 수 없음: {robot_id}")

    def send_arm_target(self, slot_id):
        if self.arm_pub:
            msg = String(); msg.data = slot_id
            self.arm_pub.publish(msg)
            print(f"로봇팔 목표 전송: {slot_id}")
    
    # OpenManipulator Start Publisher Test1
    def send_manip_start(self):
        if self.manip_start_pub:
            msg = Bool()
            msg.data = True
            self.manip_start_pub.publish(msg)
            print("▶ Published /pick_and_place/start = True")
##############################################################################
    # Jetcobot Start Publisher Test1
    def send_storage_manip_start(self):
        msg = Bool()
        msg.data = True
        self.jetco_storage_boot_pub.publish(msg)

    def send_assembly_manip_start(self):
        msg = Bool()
        msg.data = True
        self.jetco_assembly_boot_pub.publish(msg)

    def send_storage_manip_stop(self):
        msg = Bool()
        msg.data = False
        self.jetco_storage_boot_pub.publish(msg)

    def send_storage_auto_pub(self):
        msg = Int32()
        msg.data = 0
        self.jetco_mode_pub.publish(msg)

    def send_storage_manual_pub(self):
        msg = Int32()
        msg.data = 1
        self.jetco_mode_pub.publish(msg)

    def send_storage_manual_request_place(self, part_id, section, section_id):
        msg = ManualRequest()
        msg.task_id = 1
        msg.part_id = part_id
        msg.section = section
        msg.section_id = section_id
        self.jetco_storage_manual_req_pub.publish(msg)

    def send_assembly_assembly_stack_request_pub(self, module):
        msg = Int32()
        msg.data = module
        self.jetco_assembly_stack_req_pub.publish(msg)

###############################################################################

    def stop(self):
        self.running = False; self.wait()


    def trigger_manip_start(self):
        self.ros_thread.send_manip_start()
