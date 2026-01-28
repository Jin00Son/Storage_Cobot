# Storage Cobot

## 개요
스마트 팩토리 프로젝트에서 부품을 인식하여 부품 창고에 보관하는 역할을 하는 cobot의 ROS 패키지 파일

## 주요 기능

1. 관측 지점 부품들을 내부 DB에 저장 및 topic 발행
2. 관측되는 부품 중 하나 선정 후 pick하고 지정된 부품 창고 자리에 place


## 노드 구성 및 기능

- 📷 Camera Node
  - Aruco 마커 좌표 Cobot base 기준 좌표 추정
  - 위치 변화량 기반 부품 정지 상태 판단
  - 여러 부품 위치, id, 정지 상태, 정지 시간 정보를 묶어서 topic 발행
 
- 🦾 Jetcobot Node
  - Action Server: 집을 부품의 좌표와 놓아야 하는 좌표를 받아 Pick and Place 수행 (동작별 Feedback 전송)
 
- 🚦 Task Manager Node
  - 구독한 부품 정보에 기반하여 pick할 물체 선정
  - cobot에게 action 명령 보내는 주체

## Dependency




## 현버전 평가 및 한계




## 수정 및 개선 할 부분

- Required
  1. 조립용 Cobot이 Aruco Marker 인식하기 편한 방향으로 Place 하기
  2. 창고 Section 나누고 해당 section에 물체가 이미 있으면 위치 살짝 다르게 두는 알고리즘
  3. 같은 id 부품 여러개 들어왔을 때 구분하는 알고리즘

- Optional
  1. Marker 없이 물체 pose estimation
  2. jetcobot camera calibration 재측정
  3. 


