# Prophet을 이용한 암호화폐 예측 프로그램

## 설치 방법

1. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 실행 방법
```bash
python main.py
```

## 주의사항
- Upbit API를 사용하여 BTC와 ETH의 KRW 마켓 데이터를 가져옵니다
- 6개월 데이터를 기준으로 예측을 수행합니다
- 신뢰구간 상단 터치 시 매도, 하단 터치 시 매수 신호를 생성합니다
