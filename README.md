# Predicción de la calidad del agua con una red neuronal desde cero

Parcial 1 — Inteligencia Artificial · `calidad_agua.ipynb`

Análisis exploratorio (EDA) y red neuronal **7 → 16 → 8 → 2** implementada solo con NumPy: forward, backpropagation, Adam, partición 80/20 y métricas hechas a mano, sin librerías de machine learning.

## Dataset

[Water Quality Data — Kaggle](https://www.kaggle.com/datasets/supriyoain/water-quality-data/data): 2 371 muestras con salinidad, oxígeno disuelto, pH, profundidad de Secchi, profundidad y temperatura del agua, y temperatura del aire. El notebook lo descarga automáticamente con `kagglehub`.

## Cómo ejecutarlo

```bash
pip install numpy pandas matplotlib seaborn kagglehub
```

Abre el notebook en Google Colab o Jupyter y ejecuta todas las celdas. Si Kaggle pide credenciales, agrega `KAGGLE_USERNAME` y `KAGGLE_KEY` en *Secrets* de Colab.

## Contenido

1. Carga y limpieza de datos
2. EDA: cinco preguntas con hipótesis, pruebas estadísticas y gráficas
3. Preparación: partición estratificada, imputación y estandarización
4. Arquitectura: ReLU en las capas ocultas y softmax en la salida
5. Entrenamiento con Adam, regularización L2 y parada temprana
6. Evaluación y comparación con modelos lineal, Hebbiano y trivial

## Definición de la calidad

El dataset no trae una columna de calidad, así que se construye: una muestra es **apta** si tiene oxígeno disuelto ≥ 5 mg/L **y** pH entre 6.5 y 8.5. Las muestras sin esas mediciones se descartan (quedan 1 465).

## Resultados (20 % de prueba)

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Trivial (clase mayoritaria) | 0.649 | 0.393 |
| Hebbiano | 0.754 | 0.749 |
| Lineal | 0.826 | 0.814 |
| **Red neuronal 7-16-8-2** | **0.983** | **0.981** |
| Red sin oxígeno ni pH (control) | 0.724 | 0.710 |

La red supera al modelo lineal porque el pH afecta la calidad por una banda (ni muy bajo ni muy alto), algo que un modelo lineal no puede representar. El control muestra que las demás variables, sobre todo la temperatura del agua, aportan información, pero no bastan: la calidad depende principalmente del oxígeno disuelto y el pH.
