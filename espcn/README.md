# ESPCN Quantization-Aware Training

Сравнение методов квантизации для ESPCN (Super-Resolution x4).

## Результаты

### Качество и скорость на Set5 (CPU)

| Метод | PSNR (dB) | SSIM | Latency (ms) | Model Size (KB) | Compression |
|-------|-----------|------|--------------|-----------------|-------------|
| FP32 | 26.81 | 0.7569 | 0.50 | 200 | 1.0× |
| LSQ (QAT) | 26.26 | 0.7052 | 0.79 | 204 | 1.0× |
| LSQ (INT8) | 26.26 | 0.7052 | 0.87 | 32 | **6.3×** |
| PACT (QAT) | 26.06 | 0.6949 | 1.89 | 204 | 1.0× |
| PACT (INT8) | 26.06 | 0.6949 | 0.54 | 32 | **6.3×** |
| APoT (QAT) | 26.86 | 0.7506 | 1.86 | 252 | 0.8× |
| APoT (INT8) | 24.46 | 0.6811 | 1.33 | 56 | **3.6×** |
| AdaRound (PTQ) | 22.55 | 0.6167 | - | 104 | 1.9× |

**Выводы:**
- LSQ/PACT: −0.5 dB, компрессия 6.3×
- APoT: −2.4 dB при INT8, компрессия 3.6×
- AdaRound (PTQ): −4.3 dB без дообучения

## Быстрый старт

### Docker (рекомендуется)

```bash
# Сборка
docker build -t espcn-qat .

# Запуск пайплайна LSQ
docker run --rm -v $(pwd)/results:/app/results espcn-qat \
  bash run_quantization_pipeline.sh lsq

# Просмотр результатов
ls results/visualizations/
```

### Локальная установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# FP32 baseline (100 epochs, ~30 min на MPS)
python train.py  # mode="train", qat_enabled=False в config.py

# QAT LSQ (+100 epochs)
python train.py  # qat_enabled=True, qat_method="lsq"

# INT8 конверсия + тестирование
bash run_quantization_pipeline.sh lsq
```

## Структура проекта

```
espcn/
├── lsq/quant.py           # LSQ: QAConv2d, Conv2dInt
├── pact/quant.py          # PACT: с alpha регуляризацией
├── apot/quant.py          # APoT: Additive Powers-of-Two
├── adaround/ptq.py        # AdaRound PTQ
├── scripts/
│   ├── convert_qat_to_int8.py   # QAT → INT8
│   ├── test_set5.py             # PSNR/SSIM на Set5
├── run_quantization_pipeline.sh # Автопайплайн
├── config.py              # Конфигурация
└── train.py               # Обучение
```

## Воспроизведение

### 1. FP32 Baseline

```bash
# config.py: mode="train", qat_enabled=False, epochs=100
python train.py
# → results/ESPCN_x4-T91/g_last.pth.tar
```

### 2. QAT обучение

```bash
# config.py: qat_enabled=True, qat_method="lsq", epochs=100
# pretrained_model_weights_path="results/ESPCN_x4-T91/g_last.pth.tar"
python train.py
# → results/ESPCN_x4-T91-LSQ-QAT/g_best.pth.tar
```

### 3. INT8 конверсия + бенчмарк

```bash
python scripts/convert_qat_to_int8.py \
  --method lsq \
  --checkpoint results/ESPCN_x4-T91-LSQ-QAT/g_best.pth.tar
# → results/ESPCN_x4-T91-LSQ-QAT/g_best.pth_INT8/model_int8.pth.tar
```

### 4. PTQ AdaRound

```bash
python adaround/ptq.py \
  --checkpoint results/ESPCN_x4-T91/g_last.pth.tar \
  --output results/ESPCN_x4-T91-AdaRound-QAT-AdaRound
```

### Автопайплайн (всё в одной команде)

```bash
# Использует существующий FP32 чекпоинт, обучает QAT, конвертит INT8
bash run_quantization_pipeline.sh <method>  # lsq|pact|apot
```

## Данные

- **Обучение:** T91 (91 изображение, augmentation)
- **Валидация:** Set5 (5 изображений HR)
- Датасеты надо взять в опенсорсе

## Технические детали

- **Архитектура:** 3 Conv2d + PixelShuffle (r=4)
- **Обучение:** SGD, lr=1e-3, batch_size=16, ~200 epochs total
- **Метрики:** PSNR, SSIM на Y-канале (YCbCr)
- **INT8:** `Conv2dInt.from_qat()` без FX/eager calibration

## Требования

- Python 3.10+
- PyTorch 2.1+
- CUDA/MPS опционально (CPU работает)
- ~1 GB RAM для инференса

## Зависимости

```txt
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
```
