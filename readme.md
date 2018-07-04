已有可以工作的版本在 src/Deprecated 下，运行 rebuild_keras.py 即可，可能会有目录冲突的问题，使用的训练样本是在 data/training_data 下的
predict_keras.py 用 data/evaluation_data 下的数据做模型的准确性评估
continue_keras.py 载入已有模型并继续训练