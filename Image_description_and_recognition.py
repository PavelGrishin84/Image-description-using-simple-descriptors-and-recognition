# Описание изображений простыми дескрипторами и распознавание объектов на основе сравнения с эталоном
# Наборы данных:
# - изображения букв английского алфавита notMNIST (kaggle.com);
# - изображения одежды Fashion MNIST PNG (kaggle.com).

# Аналог программы Image_description_and_recognition.m на Python.

import os
import time

import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# %% 1. Загрузка изображений и вычисление дескрипторов

def find_image_files(root_dir):
    """Рекурсивный поиск файлов изображений в подпапках (аналог imageDatastore
    с \"IncludeSubfolders\", true) и получение меток классов по именам папок
    (аналог \"LabelSource\", \"foldernames\")."""
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files, labels = [], []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith(exts):
                files.append(os.path.join(dirpath, fname))
                # Метка класса — имя родительской папки
                labels.append(os.path.basename(dirpath))
    return files, labels


def imbinarize_global(img):
    """Пороговая бинаризация изображения по глобальному порогу Otsu
    (аналог imbinarize(I, 'global'))."""
    gray = np.asarray(img.convert('L'), dtype=np.uint8)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size

    # Алгоритм Оцу
    sum_all = np.dot(np.arange(256), hist)
    sum_b, w_b, best_t, best_var = 0.0, 0, 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t

    return (gray > best_t).astype(np.float32)


def compute_descriptors(file_path):
    """Вычисление дескрипторов D_1..D_4 для одного изображения.
    Возвращает tuple (D_1, D_2, D_3, D_4) либо None при ошибке чтения файла."""
    try:
        I = Image.open(file_path)
    except Exception:
        print('Ошибка чтения файла!')
        return None

    # Пороговая бинаризация изображения
    B = imbinarize_global(I)

    # Вычисление дескрипторов
    X = B.sum(axis=0)  # сумма значений пикселей по столбцам
    Y = B.sum(axis=1)  # сумма значений пикселей по строкам

    # Дескрипторы приводятся к фиксированной длине 28 (как в оригинале:
    # нескладывающиеся пары строк/столбцов обрезаются или дополняются нулями)
    def fit_len(v, n=28):
        out = np.zeros(n, dtype=np.float32)
        m = min(len(v), n)
        out[:m] = v[:m]
        return out

    Xf, Yf = fit_len(X), fit_len(Y)

    d1 = np.concatenate([Xf, Yf])          # вариант дескриптора № 1 (2*28)
    d2 = Xf - Yf                           # вариант дескриптора № 2 (28)
    d3 = np.abs(Xf - Yf)                   # вариант дескриптора № 3 (28)
    d4 = np.array([d1.mean(),              # вариант дескриптора № 4 (4)
                   d1.std(),
                   float(np.bincount(d1.astype(int)).argmax()),  # mode
                   np.median(d1)], dtype=np.float32)
    return d1, d2, d3, d4


def load_dataset(image_dir, n_jobs=-1):
    """Загрузка дескрипторов и меток для всех изображений директории
    (распараллеленный parfor-цикл из MATLAB-версии)."""
    files, labels = find_image_files(image_dir)
    print(f'Найдено изображений: {len(files)}')

    tic = time.time()
    results = Parallel(n_jobs=n_jobs)(delayed(compute_descriptors)(f) for f in files)
    toc = time.time()
    print(f'Время вычисления дескрипторов: {toc - tic:.3f} c')

    ok = [(r, lb) for r, lb in zip(results, labels) if r is not None]
    D_1 = np.array([o[0][0] for o in ok], dtype=np.float32)
    D_2 = np.array([o[0][1] for o in ok], dtype=np.float32)
    D_3 = np.array([o[0][2] for o in ok], dtype=np.float32)
    D_4 = np.array([o[0][3] for o in ok], dtype=np.float32)
    label = np.array([lb for _, lb in ok])
    return D_1, D_2, D_3, D_4, label


# %% Вспомогательные функции программы

def predict_images(cla_D, D, unique_label):
    """Вычисление модуля разности между кластерными центрами и дескрипторами
    тестовых изображений, усреднение по строкам и нахождение кластера до
    которого расстояние минимально."""
    # cla_D: (n_classes, feat), D: (n_samples, feat)
    dist = np.abs(cla_D[:, None, :] - D[None, :, :]).mean(axis=2)  # (n_classes, n_samples)
    idx = np.argmin(dist, axis=0)
    return unique_label[idx]


# %% Основная программа

def main(train_dir, test_dir=None, n_jobs=-1):
    # Загрузка тренировочных изображений (для тестирования можно передать
    # ту же директорию, как это делалось в MATLAB-версии)
    D_1, D_2, D_3, D_4, label = load_dataset(train_dir, n_jobs=n_jobs)

    # %% Вычисление центров кластеров дескрипторов обучающего набора данных

    # Уникальные метки кластеров
    unique_label = np.unique(label)

    # Для каждого класса объекта — вычисление центров кластеров
    cla_D_1 = np.stack([D_1[label == c].mean(axis=0) for c in unique_label])
    cla_D_2 = np.stack([D_2[label == c].mean(axis=0) for c in unique_label])
    cla_D_3 = np.stack([D_3[label == c].mean(axis=0) for c in unique_label])
    cla_D_4 = np.stack([D_4[label == c].mean(axis=0) for c in unique_label])

    # Тестовые данные (по умолчанию — те же, что и обучающие)
    if test_dir is None or test_dir == train_dir:
        Dt = {1: D_1, 2: D_2, 3: D_3, 4: D_4}
        label_t = label
    else:
        T1, T2, T3, T4, label_t = load_dataset(test_dir, n_jobs=n_jobs)
        Dt = {1: T1, 2: T2, 3: T3, 4: T4}

    # %% Распознавание тестовых изображений и оценка точности классификации

    centers = {1: cla_D_1, 2: cla_D_2, 3: cla_D_3, 4: cla_D_4}
    for k in (1, 2, 3, 4):
        predict = predict_images(centers[k], Dt[k], unique_label)
        Acc = np.mean(label_t == predict)
        print(f'Усредненное значение точности классификации изображений '
              f'на основе D_{k}: {Acc:f}')

        # Матрица ошибок (аналог confusionchart с column-normalized)
        cm = confusion_matrix(label_t, predict, labels=unique_label)
        cm_norm = cm.astype(float) / cm.sum(axis=0, keepdims=True).clip(min=1)
        fig = plt.figure(k)
        ConfusionMatrixDisplay(cm_norm, display_labels=unique_label).plot(cmap='Blues')
        plt.title(f'Матрица ошибок (нормализация по столбцам), D_{k}')
        plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    # Пути к наборам данных (замените на свои, как в .m-файле)
    TRAIN_DIR = r'C:\Users\gps.84\Downloads\archive\train'
    TEST_DIR = r'C:\Users\gps.84\Downloads\archive\test'
    main(TRAIN_DIR, TEST_DIR)