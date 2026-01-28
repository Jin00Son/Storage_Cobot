# Storage Cobot

## 개요



## 주요 기능

1. 관측 지점 부품들을 내부 DB에 저장 및 topic 발행
2. 관측되는 부품 중 하나 선정 후 pick하고 지정된 부품 창고 자리에 place


## 패키지 구성

  ├── jetcobot_pkg
  │   ├── __init__.py
  │   ├── camera_node.py
  │   ├── jetcobot_node.py
  │   ├── task_manager_node.py
  │   └── utils (유틸 함수 저장)
  │       ├── camera_utils.py
  │       ├── cobot_utils.py
  │       ├── __init__.py
  │       └──   ready_to_pick.py
  |   ├── calib_data (calibration 정보 저장)
  |   ├── launch (ROS launch 파일 저장)
  |      └── storage_launch.py

  ├── jetcobot_interfaces  (ROS 통신 정의 패키지)
  │   ├── action
  │   │   └── PickAndPlace.action
  │   ├── msg
  │   │   ├── PartArray.msg
  │   │   └── Part.msg
  │   └── srv
  │       ├── ManualPick.srv
  │       └── SetTaskMode.srv




## 노드 구성
- Camera Node
- Jetcobot Node
- Task Manager Node

