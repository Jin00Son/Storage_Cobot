#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional, Tuple

from rclpy.node import Node
from rclpy.action import ActionClient

from jetcobot_interfaces.action import PickAndPlace


class PickAndPlaceClient:
    """
    ✅ ActionClient 유틸 클래스 (feedback/result 콜백을 클래스 내부에서 모두 처리)
    - send_goal(pick_coords, place_coords) 만 호출하면 됨
    - 결과는 consume_done()으로 가져오거나, is_busy()로 상태 확인 가능
    """

    def __init__(self, node: Node, action_name: str = "/pickandplace", wait_server_timeout_sec: float = 1.0):
        self._node = node
        self._action_name = action_name
        self._wait_server_timeout_sec = float(wait_server_timeout_sec)

        self._client = ActionClient(node, PickAndPlace, action_name)

        # 내부 상태
        self._busy: bool = False
        self._goal_handle = None

        self._done_ready: bool = False
        self._done_success: bool = False
        self._done_message: str = ""

    # -------------------------
    # Public API
    # -------------------------
    def is_busy(self) -> bool:
        return bool(self._busy)

    def action_done(self) -> Optional[Tuple[bool, str]]:
        """
        ✅ 완료 결과를 '한 번만' 꺼내감
        return:
          - None: 아직 완료 아님
          - (success, message): 완료됨
        """
        if not self._done_ready:
            return None
        out = (bool(self._done_success), str(self._done_message))

        # consume이므로 초기화
        self._done_ready = False
        self._done_success = False
        self._done_message = ""

        return out

    def send_goal(self, pick_coords: List[float], place_coords: List[float]) -> bool:
        """
        ✅ 파라미터 최소화: pick/place만 받음
        return:
          - True: goal 전송 시작
          - False: 서버 없음/입력 오류/이미 busy
        """
        if self._busy:
            self._node.get_logger().warn("[P&P] send_goal ignored: already busy")
            return False

        if pick_coords is None or place_coords is None:
            self._node.get_logger().error("[P&P] send_goal failed: pick/place is None")
            return False

        if len(pick_coords) != 6 or len(place_coords) != 6:
            self._node.get_logger().error("[P&P] send_goal failed: pick/place must be length 6")
            return False

        if not self._client.wait_for_server(timeout_sec=self._wait_server_timeout_sec):
            self._node.get_logger().error(f"[P&P] action server not available: {self._action_name}")
            return False

        goal = PickAndPlace.Goal()
        goal.pick_coords = [float(x) for x in pick_coords]
        goal.place_coords = [float(x) for x in place_coords]

        # 내부 상태 초기화
        self._busy = True
        self._done_ready = False
        self._done_success = False
        self._done_message = ""
        self._last_feedback_progress = 0.0
        self._last_feedback_state = ""

        self._node.get_logger().info(f"[P&P] send_goal -> Pick and Place Goal Sent!")

        send_future = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        send_future.add_done_callback(self._goal_response_cb)
        return True

    # -------------------------
    # Internal callbacks
    # -------------------------
    def _feedback_cb(self, fb_msg):
        fb = fb_msg.feedback
        self._last_feedback_progress = float(fb.progress)
        self._last_feedback_state = str(fb.state)

        # ✅ 기본 로깅은 클래스 내부에서 처리
        self._node.get_logger().info(f"[P&P FB] {self._last_feedback_progress:.1f}% {self._last_feedback_state}")

    def _goal_response_cb(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self._finish(False, f"goal_response exception: {e}", None)
            return

        if not goal_handle.accepted:
            self._finish(False, "Goal rejected", None)
            return

        self._node.get_logger().info("[P&P] Goal accepted ✅")
        self._goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        try:
            wrapped = future.result()
            result = wrapped.result
        except Exception as e:
            self._finish(False, f"result exception: {e}", None)
            return

        success = bool(getattr(result, "success", False))
        message = str(getattr(result, "message", ""))

        self._finish(success, message)

    def _finish(self, success: bool, message: str):
        # 완료 저장
        self._done_ready = True
        self._done_success = bool(success)
        self._done_message = str(message)

        # busy 해제
        self._busy = False
        self._goal_handle = None

        self._node.get_logger().info(f"[P&P DONE] success={self._done_success} msg={self._done_message}")
