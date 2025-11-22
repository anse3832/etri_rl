# etri_rl
Graduate course (Advanced Reinforcement Learning) term project

아래의 메뉴얼대로 순서대로 진행하시면 됩니다.

## 0. Preresuites
conda 가상환경에서 실행해야 하기 때문에 anaconda3 설치 필수<br>
git으로 코드 다운을 해야 하므로 git이 동작해야 함

## 1. 코드 다운로드 
```
git clone https://github.com/anse3832/etri_rl
cd etri_rl
```

## 2. conda 가상환경 생성 및 접속
conda 가상환경 생성시 pytorch를 포함한 다양한 패키지 설치를 하기 때문에 꽤 오랜 시간이 걸립니다<br>
중지된 것처럼 보여도 패키지 설치를 하는 중이기 때문에 기다리시면 설치가 완료됩니다<br>
```
conda env create -n <가상환경_이름> -f environment.yml
conda activate <가상환경_이름>
```

## 3. 모델 zip 파일 다운로드
아래의 google drive 링크에서 다운받은 "best_model_QRDQN.zip" 파일을<br> 
"1. 코드 다운로드" 과정에서 생성된 "etri_rl" 폴더 안으로 이동<br>
https://drive.google.com/file/d/138_4vDAVEY5XTMqmcmR56WlSflOq6Zpv/view?usp=drive_link
<img width="500" alt="그림8" src="https://github.com/user-attachments/assets/0477e01f-fcf8-403c-9f7f-aec0d5a10216" />


## 4. 코드 실행
```
python spaceinvader.py
```
spaceinvader.py 안의 경로를 반드시 맞춰줄 것
<img width="500" height="715" alt="그림7" src="https://github.com/user-attachments/assets/9290ac69-80eb-438c-b7f9-1866afc34e90" />

## 5. 실행 결과
아래의 화면이 생성되면 성공
(총 10번의 episode 실행)
<img width="500" alt="스크린샷 2025-11-21 081734" src="https://github.com/user-attachments/assets/09e4bf7e-6a64-4025-af4b-99709fcf4136" />
