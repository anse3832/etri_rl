# etri_rl
Graduate course (Advanced Reinforcement Learning) term project

아래의 메뉴얼대로 순서대로 진행하시면 됩니다.

## 1. 코드 다운로드 
```
git clone https://github.com/anse3832/etri_rl
cd etri_rl
```

## 2. conda 가상환경 생성 및 접속
conda 가상환경 생성시, pytorch 등 다양한 패키지 설치로 인해 시간이 오래 걸림
```
conda env create -n <가상환경_이름> -f environment.yml
conda activate <가상환경_이름>
```

## 3. 모델 zip 파일 다운로드
아래의 링크에서 다운받은 "best_model_QRDQN.zip" 파일을<br> 
"1. 코드 다운로드" 과정에서 생성된 "etri_rl" 폴더 안으로 이동
https://drive.google.com/file/d/138_4vDAVEY5XTMqmcmR56WlSflOq6Zpv/view?usp=drive_link

## 4. 코드 실행
```
python spaceinvader.py
```
spaceinvader.py 안의 경로를 반드시 맞춰줄 것
<img width="2270" height="1591" alt="그림7" src="https://github.com/user-attachments/assets/9290ac69-80eb-438c-b7f9-1866afc34e90" />

## 5. 실행 결과
아래의 화면이 생성되면 성공
(총 10번의 episode 실행)
<img width="854" height="670" alt="스크린샷 2025-11-21 081734" src="https://github.com/user-attachments/assets/09e4bf7e-6a64-4025-af4b-99709fcf4136" />
