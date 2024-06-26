# 이미지 기반 레시피 추천 Project
---
### 주제
- 음식 이미지를 딥러닝 모델을 사용하여 판별하여 음식 명칭, 간단한 레시피, 인기 많은 레시피 정보 제공
---
### 프로젝트 수행 방안
1. 데이터 수집
   - 한국 음식 이미지 (출처 : Ai-Hub) 150종(음식당 약 1천장)의 데이터
     (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=79)
   - 웹 크롤링 : 만개의 레시피 사이트에서 음식명, 재료, 레시피, 조회수 등의 정보 수집
   - 
2. 이미지 전처리
   - 이미지 : 정사각형의 이미지로 만들기 위해 짧은 변에 맞춰 양 끝을 잘라서 준비
   - 텍스트 : 자연어 분석을 위한 전처리

3. 데이터 분석
   - VGG16, Inception V3, MobileNet V2 모델을 사용하여 분석
   - 가장 정확도가 높았던 Inception 모델 fine tuning

4. 서비스 구현 및 시각화
   - Python, Flask를 활용한 웹서비스 개발
   - 가장 정확도가 높았던 모델로 사용자가 업로드한 사진을 판별
   - 판별된 사진으로 음식명, 간단한 레시피, 인기 많은 레시피 정보 제공
  
