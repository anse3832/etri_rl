# etri_rl
Graduate course (Advanced Reinforcement Learning) term project

## 1. 코드 다운로드 
```
git clone https://github.com/anse3832/etri_rl
cd etri_rl
```

## 2. conda 가상환경 생성 및 접속
conda 가상환경 생성시, pytorch 등 다양한 패키지 설치로 인해 오랜 시간이 걸림
```
conda env create -n <가상환경_이름> -f environment.yml
conda activate <가상환경_이름>
```

## 3. 코드 실행
```
python spaceinvader.py
```
아래와 같은 화면이 생성되면 성공
(총 10번의 episode 실행)
