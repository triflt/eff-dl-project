Скрипт для воспроизведения результатов:
```bash
docker build -t eff-lstm .
docker run -v "$(pwd)/artifacts_docker:/app/artifacts" -v "$(pwd)/plots_docker:/app/plots" eff-lstm
```

LSTM

Датасет:
* SMS Spam Collection Dataset
* Бинарная классификация
* 5500 семплов
* Средняя длина текста в семпле - 100 символов

Модель:
* Однонаправленная LSTM
* 2 слоя
* Размерности слоя эмбеддингов и скрытый слой 96
* Размер словаря 3500 токенов

PACT:
![losses](plots/pact/pact_losses.png)
![metrics](plots/pact/pact_metrics.png)

LSQ:
![losses](plots/lsq/lsq_losses.png)
![metrics](plots/lsq/lsq_metrics.png)

APoT:
![losses](plots/apot/apot_losses.png)
![metrics](plots/apot/apot_metrics.png)

AdaRound:
![losses](plots/adaround/adaround_losses.png)
![metrics](plots/adaround/adaround_metrics.png)

EfficientQAT:
![losses](plots/efficientqat/efficientqat_losses.png)
![metrics](plots/efficientqat/efficientqat_metrics.png)

Overall:
![train](plots/timings/train.png)
![infer](plots/timings/infer.png)
![qa](quantized_infer/timings/quantized_infer.png)

Выводы:
* Квантизация занимает много времени замедляя train и inference
* int8 умножения в torch написаны не так эффективно как fp32
* Для LSQ легче найти нужный конфиг не приводящий к ухудшению метрик
