# Sprint 14 BikeSouth

Проект прогнозирует почасовой спрос на аренду велосипедов для BikeSouth.

В репозитории:

- `Sprint_14_proj.ipynb` - основная тетрадка с анализом, обучением и выводами.
- `bike_feature_engineer.py` - модуль с кастомным трансформером `BikeFeatureEngineer`.
- `best_dtree_pipeline.joblib` - сохраненный pipeline лучшей модели из тетрадки.
- `baseline_linear_regression_pipeline.joblib` - baseline-модель из задания.
- `ds_s14_train_data.csv` и `ds_s14_test_data.csv` - обучающая и тестовая выборки.
- `Project_explanation.md` - исходное техническое задание.
- `requirements.txt` - версии ключевых библиотек, соответствующие выполненной тетрадке и pipeline.

Основной результат: DecisionTreeRegressor после Optuna улучшает baseline линейной регрессии на тестовой выборке.

Для повторного использования pipeline сначала импортируйте трансформер:

```python
from bike_feature_engineer import BikeFeatureEngineer
```
