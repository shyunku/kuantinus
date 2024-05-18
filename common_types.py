import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import math


class PlotLearning(Callback):
    def __init__(self, model_idx=0):
        self.model_idx = model_idx
        
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.mse = []
        self.val_mse = []
        plt.ion()  # 인터랙티브 모드 활성화
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 12))  # 2개의 플롯 준비
        self.fig.canvas.manager.window.setWindowTitle(f'Train #{self.model_idx}')
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.mse.append(math.sqrt(logs.get('mse')))
        self.val_mse.append(math.sqrt(logs.get('val_mse')))
        self.i += 1

        self.ax1.clear()  # 첫 번째 축의 내용을 클리어
        self.ax1.set_yscale('log', base=2)
        self.ax1.plot(self.x, self.losses, label="Training Loss")
        self.ax1.plot(self.x, self.val_losses, label="Validation Loss")
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training and Validation Loss')
        self.ax1.grid()
        self.ax1.legend()
        self.ax1.annotate(f'{self.losses[-1]:.4f}', xy=(self.x[-1], self.losses[-1]), textcoords="offset points", xytext=(0,10), ha='center')

        self.ax2.clear()  # 두 번째 축의 내용을 클리어
        self.ax2.set_yscale('log', base=2)
        self.ax2.plot(self.x, self.mse, label="Training RMSE")
        self.ax2.plot(self.x, self.val_mse, label="Validation RMSE")
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('RMSE')
        self.ax2.set_title('Training and Validation RMSE')
        self.ax2.grid()
        self.ax2.legend()
        self.ax2.annotate(f'{self.mse[-1]:.4f}', xy=(self.x[-1], self.mse[-1]), textcoords="offset points", xytext=(0,10), ha='center')

        plt.pause(0.2)  # 그래프를 업데이트하고 잠시 대기

    def on_train_end(self, logs={}):
        plt.ioff()  # 인터랙티브 모드 비활성화
        plt.show()  # 훈련이 끝난 후 창 유지


class DynamicLRScheduler(Callback):
    def __init__(self, base_lr=0.001):
        super().__init__()
        self.base_lr = base_lr
        self.iteration = 0
    
    def on_epoch_end(self, epoch, logs=None):
        self.iteration += 1
        new_lr = self.base_lr * 0.99 ** self.iteration
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)