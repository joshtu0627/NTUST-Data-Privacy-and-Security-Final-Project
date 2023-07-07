# 使用說明

## data

colorado_original.csv => 原始資料集
colorado_original_INCWAGE_dropped.csv => 去除有不合理的INCWAGE欄位值資料的資料集

synthetic_data_INCWAGE_dropped.csv => ε為1來生成的synthetic data
synthetic_data_INCWAGE_dropped(0.5).csv => ε為0.5來生成的synthetic data
synthetic_data_INCWAGE_dropped(0.2).csv => ε為0.2來生成的synthetic data

## code

k-Anonymity_data.py => 生成k-Anonymity資料集，並以此資料集訓練模型並生成實驗結果的code
l-Diversity_data.py => 生成l-diversity資料集，並以此資料集訓練模型並生成實驗結果的code
t-Closeness_data.py => 生成t-Closeness資料集，並以此資料集訓練模型並生成實驗結果的code

synthetic_data_model_building.py => 以synthetic data來生成模型的code
synthetic_data_model_testing.py => 以synthetic data所訓練的模型來預測真實資料，並產生實驗結果的code

## model

model/k-Anonymity_data.h5 => k-Anonymity_data所產生的模型
model/l-Diversity_data.h5 => l-Diversity_data所產生的模型
model/t-Closeness_data.h5 => t-Closeness_data所產生的模型

model/synthetic_data.h5(ε值) => synthetic_data_model_building.py所產生的模型
