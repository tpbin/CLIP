# CLIP: Contrastive Language–Image Pre‑training

> **논문**: *Learning Transferable Visual Models From Natural Language Supervision* (Radford *et al.*, ICML 2021)
> **코드**: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

---

## 1. 등장 배경

* 기존 대규모 시각 모델은 **수십만 ~ 수백만 개의 **정제 라벨(curated labels)**** 의존 → 라벨 비용과 도메인 제약
* 웹에는 "이미지 + 캡션" 형태의 **자연어 감독(natural-language supervision)** 이 방대하게 존재
* 질문: *자연어만으로 강력한 시각 표현을 학습할 수 있을까?* —> **CLIP** 제안

## 2. 핵심 기여

1. **Dual‑Encoder 대조 학습(contrastive learning)**: 이미지 · 텍스트 인코더를 동시에 학습해 공통 임베딩 공간 정렬
2. **웹‑스케일 데이터(web‑scale data)**: 4억 (image,text) 쌍(WIT) → 라벨 없이도 표현 학습
3. **제로샷 전이**: 텍스트 프롬프트만으로 30+ 태스크를 라벨 없이 해결

---

## 3. Model Architecture

![image](https://github.com/user-attachments/assets/e145ed72-0afd-49e1-a6fb-e3abc31d3b5b)
<sub>▲ 논문 Figure 1: (1) 대조 사전학습 → (2) 텍스트‑기반 분류기 생성 → (3) 제로샷 예측 흐름</sub>

### (1) Contrastive Pre‑training

| 단계             | 설명                                                                                |
| -------------- | --------------------------------------------------------------------------------- |
| **텍스트 인코딩**    | 캡션(예: “Pepper the aussie pup”)을 BPE 토크나이즈 → Transformer 텍스트 인코더가 임베딩 **T₁…Tₙ** 생성 |
| **이미지 인코딩**    | 대응 이미지가 ResNet·ViT 기반 이미지 인코더를 거쳐 임베딩 **I₁…Iₙ** 생성                                |
| **유사도 행렬**     | 한 배치 *N*쌍으로 **N × N** 코사인 유사도 테이블 생성                                              |
| **InfoNCE 손실** | 대각선(올바른 쌍)을 높이고 나머지를 낮추도록 대조 교차 엔트로피 최적화                                          |

### (2) Create Dataset Classifier from Label Text

| 단계          | 설명                                                                    |
| ----------- | --------------------------------------------------------------------- |
| **레이블 문장화** | 데이터셋 레이블(plane, car, dog …)을 템플릿 *"A photo of a {object}."* 에 삽입해 문장화 |
| **텍스트 임베딩** | 문장들을 텍스트 인코더에 통과 → **클래스 프로토타입 벡터**로 저장                               |

### (3) Use for Zero‑shot Prediction

| 단계          | 설명                                       |
| ----------- | ---------------------------------------- |
| **이미지 임베딩** | 입력 이미지를 이미지 인코더에 통과해 임베딩 **I** 획득        |
| **유사도 계산**  | **I**와 모든 클래스 프로토타입 간 코사인 유사도 계산         |
| **최종 예측**   | 가장 높은 유사도의 레이블을 출력 → **라벨 0** 상태에서 분류 완료 |

---

### ▶ What is Zero‑Shot Transfer?

> **“모델을 특정 태스크용으로 추가 학습(fine‑tuning)하지 않고도, 전혀 본 적 없는 데이터셋·클래스에 바로 적용해 성능을 내는 능력.”**\
> *shot* = 라벨이 달린 학습 샘플 수, **zero‑shot** ⇒ 라벨 **0**.

**①  Why it matters**

- **데이터 효율성**: 새로운 과제에 라벨을 따로 붙이지 않아도 즉시 사용 가능
- **범용성**: 수십 개 태스크를 하나의 사전학습 모델로 해결 → 유지 비용↓

**②  CLIP Zero‑Shot Flow**

1. **Label → Prompt**: *“dog” → “A photo of a dog.”*
2. **Prompt Embedding**: 텍스트 인코더 → 클래스 프로토타입 벡터
3. **Image Embedding**: 입력 이미지 → 이미지 벡터
4. **Similarity & Argmax**: 코사인 유사도 계산 → 최고 점수가 예측 라벨

> ViT‑L/14\@336 모델은 이 방식만으로 ImageNet Top‑1 \*\*76 %\*\*를 달성

---

## 4. Key Advantages

* **Label‑free Transfer** : 데이터셋 라벨 없이도 프롬프트만으로 분류 가능
* **멀티모달 확장성** : 텍스트 질의 → 이미지 검색·OCR·행동 인식 등 광범위 적용
* **스케일 효율** : 동일 연산 대비 ResNet > ViT, 대조 학습이 예측 학습보다 4× 빠른 전이 효과

---

## 5. Results & Comparison

### 5.1 Zero-Shot Headline

| Dataset  | Visual N-Grams (zero-shot) | **CLIP (zero-shot)** |
|----------|---------------------------|----------------------|
| aYahoo   | 72.4 % | **98.4 %** |
| ImageNet | 11.5 % | **76.2 %** |
| SUN      | 23.0 % | **58.5 %** |

> 기존 zero-shot 방법인 Visual N-Grams와 비교했을 때, CLIP은 세 데이터셋 모두에서 두 자릿수 이상을 기록<br>
> ImageNet에서는 **+64.7 %p**의 극적인 향상<br>
> 즉, "라벨 0개" 상황에서도 CLIP이 과거 방식의 한계를 단숨에 뛰어넘는다는 점을 강조

### 5.2 Scaling Results (Representative Datasets)

| Dataset      | Supervised RN50 | **CLIP RN50** | **CLIP ViT-B/16** | **CLIP ViT-L/14 @ 336** |
|--------------|-----------------|---------------|-------------------|-------------------------|
| ImageNet     | 76.2 % | 63.3 % | 72.0 % | **76.2 %** |
| CIFAR-100    | 79.8 % | 71.9 % | 80.5 % | **82.3 %** |
| Oxford Pets  | 93.5 % | 94.5 % | 97.3 % | **98.0 %** |

> CLIP은 모델 크기가 커질수록 성능이 꾸준히 상승<br>
> ViT-L/14 @ 336 px 버전은 **ImageNet-supervised ResNet-50**과 동일 수준(76 %)에 도달<br>
> 소규모·세분화 데이터셋(CIFAR-100, Oxford Pets)에서도 추가 이득을 제공  
> 이는 "모델 ↗ + 데이터 ↗ → 성능 ↗"라는 스케일링 법칙이 멀티모달 대조 학습에서도 그대로 적용됨을 시사

<br>
** Figure 7. Zero-shot CLIP is much more robust to distribution shift than standard ImageNet models.<br>

<img src="https://github.com/user-attachments/assets/efa9837f-318d-4982-b066-5c3388c5aca6" width="350">

　　<sub>▲ 배포 변화에 대한 zero‑shot 견고성: CLIP 사용 시 오류 갭 75% 감소</sub>
<br><br>
> **Few‑shot**에서도 CLIP 표현 위 간단한 선형 분류기는 기존 SOTA 대비 최대 10× 빠르게 수렴하고 더 높은 정확도 달성

---

## 6. Limitations & Bias

* **Prompt 민감도** : 단어·어순·구두점 표현에 결과 급변 → prompt engineering 필요
* **데이터 편향** : 웹 캡션 기반이라 사회적 편향·부적절 언어 포함 가능
* **해상도 제약** : 224²/336² 범위에서 학습 → 더 높은 해상도(예: 512²) 입력 시 성능 저하
* **계산 비용** : 256 × V100, 18 일 학습 — 소규모 연구 그룹엔 부담
