## Import Semua Library


```python
!pip install split-folders --quiet
```


```python
import os
import glob
import random
import shutil
import pathlib
import splitfolders
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, Callback)


# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow :", tf.__version__)
print("Keras      :", tf.keras.__version__)
print("GPU aktif  :", tf.config.list_physical_devices('GPU'))
```

    TensorFlow : 2.20.0
    Keras      : 3.13.2
    GPU aktif  : [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


## Data Preparation


```python
dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")
print("Lokasi dataset:", dataset_path)

# Cari folder yang berisi sub-folder kelas ('raw-img')
kandidat = glob.glob(os.path.join(dataset_path, "**", "raw-img"), recursive=True)
SOURCE_DIR = kandidat[0] if kandidat else dataset_path
print("Direktori sumber kelas:", SOURCE_DIR)
print("Kelas tersedia (nama asli/Italia):", sorted(os.listdir(SOURCE_DIR)))
```

    Using Colab cache for faster access to the 'animals10' dataset.
    Lokasi dataset: /kaggle/input/animals10
    Direktori sumber kelas: /kaggle/input/animals10/raw-img
    Kelas tersedia (nama asli/Italia): ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']



```python
# =========================================================
TRANSLATE = {
    "cane": "anjing", "cavallo": "kuda", "elefante": "gajah",
    "farfalla": "kupu_kupu", "gallina": "ayam", "gatto": "kucing",
    "mucca": "sapi", "pecora": "domba", "ragno": "laba_laba",
    "scoiattolo": "tupai",
}
# Total ~14.900 gambar
SELECTED_CLASSES = ["cane", "ragno", "gallina", "farfalla"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

WORK_DIR = "/content/dataset_kerja"
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
os.makedirs(WORK_DIR)

total_disalin, total_dilewati = 0, 0
for kelas_it in SELECTED_CLASSES:
    src = os.path.join(SOURCE_DIR, kelas_it)
    dst = os.path.join(WORK_DIR, TRANSLATE[kelas_it])
    os.makedirs(dst, exist_ok=True)
    for berkas in os.listdir(src):
        if pathlib.Path(berkas).suffix.lower() not in VALID_EXT:
            total_dilewati += 1
            continue
        try:
            with Image.open(os.path.join(src, berkas)) as im:
                im.verify()
            shutil.copy2(os.path.join(src, berkas), os.path.join(dst, berkas))
            total_disalin += 1
        except Exception:
            total_dilewati += 1

print(f"Total gambar valid disalin : {total_disalin}")
print(f"Total berkas dilewati      : {total_dilewati} (rusak / bukan gambar)")
print("Kelas final                :", sorted(os.listdir(WORK_DIR)))
```

    Total gambar valid disalin : 14894
    Total berkas dilewati      : 0 (rusak / bukan gambar)
    Kelas final                : ['anjing', 'ayam', 'kupu_kupu', 'laba_laba']



```python
# =========================================================
def ringkas_dataset(direktori):
    ukuran_unik = set()
    total = 0
    distribusi = {}
    for sub in sorted(os.listdir(direktori)):
        sub_path = os.path.join(direktori, sub)
        if not os.path.isdir(sub_path):
            continue
        berkas = os.listdir(sub_path)
        distribusi[sub] = len(berkas)
        total += len(berkas)
        for f in berkas:
            with Image.open(os.path.join(sub_path, f)) as img:
                ukuran_unik.add(img.size)
    return total, distribusi, ukuran_unik

total_gambar, distribusi, ukuran_unik = ringkas_dataset(WORK_DIR)
print("Distribusi gambar per kelas:")
for k, v in distribusi.items():
    print(f"  - {k:12s}: {v}")
print(f"\nTotal gambar          : {total_gambar}")
print(f"Jumlah resolusi unik  : {len(ukuran_unik)}")
print("Contoh resolusi asli  :", list(ukuran_unik)[:8])

assert total_gambar >= 10000, "Total gambar harus >= 10.000."
assert len(distribusi) >= 3,  "Jumlah kelas harus >= 3."
assert len(ukuran_unik) > 1,  "Resolusi asli harus beragam (saran)."
print("\nMemenuhi: >= 10.000 gambar, >= 3 kelas, resolusi beragam.")

# Visualisasi distribusi kelas
plt.figure(figsize=(8, 4))
plt.bar(list(distribusi.keys()), list(distribusi.values()), color="steelblue")
plt.title("Distribusi Jumlah Gambar per Kelas")
plt.ylabel("Jumlah gambar")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```

    Distribusi gambar per kelas:
      - anjing      : 4863
      - ayam        : 3098
      - kupu_kupu   : 2112
      - laba_laba   : 4821
    
    Total gambar          : 14894
    Jumlah resolusi unik  : 724
    Contoh resolusi asli  : [(449, 640), (640, 320), (300, 270), (640, 329), (300, 279), (181, 300), (640, 576), (278, 251)]
    
    Memenuhi: >= 10.000 gambar, >= 3 kelas, resolusi beragam.



    
![png](Notebook_DL_files/Notebook_DL_7_1.png)
    



```python
# =========================================================
kelas_list = sorted(os.listdir(WORK_DIR))
plt.figure(figsize=(12, 3 * len(kelas_list)))
for i, kelas in enumerate(kelas_list):
    contoh = os.listdir(os.path.join(WORK_DIR, kelas))[:4]
    for j, berkas in enumerate(contoh):
        ax = plt.subplot(len(kelas_list), 4, i * 4 + j + 1)
        img = Image.open(os.path.join(WORK_DIR, kelas, berkas))
        ax.imshow(img)
        ax.set_title(f"{kelas}\n{img.size}", fontsize=9)
        ax.axis("off")
plt.tight_layout()
plt.show()
```


    
![png](Notebook_DL_files/Notebook_DL_8_0.png)
    


### Data Preprocessing

#### Split Dataset


```python
OUTPUT_SPLIT = "/content/dataset_split"
if os.path.exists(OUTPUT_SPLIT):
    shutil.rmtree(OUTPUT_SPLIT)

splitfolders.ratio(WORK_DIR, output=OUTPUT_SPLIT, seed=SEED, ratio=(0.8, 0.1, 0.1))
print("Struktur hasil split:", os.listdir(OUTPUT_SPLIT))

# ---- Parameter citra ----
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

def muat_dataset(subset, acak):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(OUTPUT_SPLIT, subset),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=acak,
        seed=SEED,
    )

train_ds = muat_dataset("train", acak=True)
val_ds   = muat_dataset("val",   acak=False)
test_ds  = muat_dataset("test",  acak=False)

class_names = train_ds.class_names
print("Label kelas (urutan indeks):", class_names)

# Data validation & test
AUTOTUNE = tf.data.AUTOTUNE
augmentasi = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.10),
], name="augmentasi_data")

train_ds = (train_ds
            .map(lambda x, y: (augmentasi(x, training=True), y),
                 num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))
val_ds  = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)
```

    Copying files: 14894 files [00:02, 5952.79 files/s]


    Struktur hasil split: ['test', 'train', 'val']
    Found 11913 files belonging to 4 classes.
    Found 1488 files belonging to 4 classes.
    Found 1493 files belonging to 4 classes.
    Label kelas (urutan indeks): ['anjing', 'ayam', 'kupu_kupu', 'laba_laba']


## Modelling


```python
jumlah_kelas = len(class_names)

model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.30),

    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.30),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.50),
    layers.Dense(jumlah_kelas, activation="softmax"),
], name="cnn_klasifikasi_hewan")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "cnn_klasifikasi_hewan"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ rescaling (<span style="color: #0087ff; text-decoration-color: #0087ff">Rescaling</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">128</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">9,248</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">160</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">128</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_3           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">80</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_4           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │           <span style="color: #00af00; text-decoration-color: #00af00">512</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_5           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │           <span style="color: #00af00; text-decoration-color: #00af00">512</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_6           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">65,792</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_7           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">1,028</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">652,836</span> (2.49 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">650,916</span> (2.48 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,920</span> (7.50 KB)
</pre>




```python
class StopPada95(Callback):
    def __init__(self, target=0.95):
        super().__init__()
        self.target = target
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (logs.get("accuracy", 0) >= self.target and
                logs.get("val_accuracy", 0) >= self.target):
            print(f"\nAkurasi train & validation >= {self.target:.0%}. "
                  f"Pelatihan dihentikan.")
            self.model.stop_training = True

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint("/content/model_terbaik.keras", monitor="val_accuracy", save_best_only=True),
    StopPada95(0.95),
]

EPOCHS = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)
```

    Epoch 1/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m113s[0m 239ms/step - accuracy: 0.4608 - loss: 1.4119 - val_accuracy: 0.4140 - val_loss: 1.3403 - learning_rate: 0.0010
    Epoch 2/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m80s[0m 214ms/step - accuracy: 0.6277 - loss: 0.9540 - val_accuracy: 0.4341 - val_loss: 1.4875 - learning_rate: 0.0010
    Epoch 3/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m76s[0m 202ms/step - accuracy: 0.7025 - loss: 0.7760 - val_accuracy: 0.5370 - val_loss: 1.2153 - learning_rate: 0.0010
    Epoch 4/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 198ms/step - accuracy: 0.7460 - loss: 0.6671 - val_accuracy: 0.6344 - val_loss: 0.9026 - learning_rate: 0.0010
    Epoch 5/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.7808 - loss: 0.5984 - val_accuracy: 0.6465 - val_loss: 0.9285 - learning_rate: 0.0010
    Epoch 6/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 192ms/step - accuracy: 0.7969 - loss: 0.5560 - val_accuracy: 0.6337 - val_loss: 1.2942 - learning_rate: 0.0010
    Epoch 7/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.8034 - loss: 0.5289 - val_accuracy: 0.6956 - val_loss: 0.7880 - learning_rate: 0.0010
    Epoch 8/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.8182 - loss: 0.5019 - val_accuracy: 0.6848 - val_loss: 0.8120 - learning_rate: 0.0010
    Epoch 9/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 197ms/step - accuracy: 0.8268 - loss: 0.4745 - val_accuracy: 0.5612 - val_loss: 1.3515 - learning_rate: 0.0010
    Epoch 10/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 198ms/step - accuracy: 0.8311 - loss: 0.4601 - val_accuracy: 0.7231 - val_loss: 0.8516 - learning_rate: 0.0010
    Epoch 11/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.8356 - loss: 0.4513 - val_accuracy: 0.7224 - val_loss: 0.7506 - learning_rate: 0.0010
    Epoch 12/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 195ms/step - accuracy: 0.8424 - loss: 0.4338 - val_accuracy: 0.6169 - val_loss: 1.1176 - learning_rate: 0.0010
    Epoch 13/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.8573 - loss: 0.4081 - val_accuracy: 0.7285 - val_loss: 0.7657 - learning_rate: 0.0010
    Epoch 14/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 196ms/step - accuracy: 0.8526 - loss: 0.4191 - val_accuracy: 0.7480 - val_loss: 0.8540 - learning_rate: 0.0010
    Epoch 15/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.8592 - loss: 0.3976 - val_accuracy: 0.8017 - val_loss: 0.5308 - learning_rate: 0.0010
    Epoch 16/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 193ms/step - accuracy: 0.8591 - loss: 0.3925 - val_accuracy: 0.6284 - val_loss: 0.9620 - learning_rate: 0.0010
    Epoch 17/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 197ms/step - accuracy: 0.8686 - loss: 0.3743 - val_accuracy: 0.7749 - val_loss: 0.6969 - learning_rate: 0.0010
    Epoch 18/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 197ms/step - accuracy: 0.8705 - loss: 0.3662 - val_accuracy: 0.8353 - val_loss: 0.4822 - learning_rate: 0.0010
    Epoch 19/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.8774 - loss: 0.3419 - val_accuracy: 0.7836 - val_loss: 0.6695 - learning_rate: 0.0010
    Epoch 20/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.8774 - loss: 0.3424 - val_accuracy: 0.8091 - val_loss: 0.5512 - learning_rate: 0.0010
    Epoch 21/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.8837 - loss: 0.3344 - val_accuracy: 0.6989 - val_loss: 0.7867 - learning_rate: 0.0010
    Epoch 22/50
    [1m372/373[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 191ms/step - accuracy: 0.8855 - loss: 0.3288
    Epoch 22: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 194ms/step - accuracy: 0.8863 - loss: 0.3202 - val_accuracy: 0.8239 - val_loss: 0.5223 - learning_rate: 0.0010
    Epoch 23/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 191ms/step - accuracy: 0.8995 - loss: 0.2830 - val_accuracy: 0.8320 - val_loss: 0.4694 - learning_rate: 5.0000e-04
    Epoch 24/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.9037 - loss: 0.2755 - val_accuracy: 0.7917 - val_loss: 0.6186 - learning_rate: 5.0000e-04
    Epoch 25/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 196ms/step - accuracy: 0.9066 - loss: 0.2631 - val_accuracy: 0.8663 - val_loss: 0.3926 - learning_rate: 5.0000e-04
    Epoch 26/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 194ms/step - accuracy: 0.9067 - loss: 0.2574 - val_accuracy: 0.8562 - val_loss: 0.4045 - learning_rate: 5.0000e-04
    Epoch 27/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 199ms/step - accuracy: 0.9111 - loss: 0.2517 - val_accuracy: 0.8656 - val_loss: 0.3957 - learning_rate: 5.0000e-04
    Epoch 28/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 197ms/step - accuracy: 0.9112 - loss: 0.2515 - val_accuracy: 0.8622 - val_loss: 0.4573 - learning_rate: 5.0000e-04
    Epoch 29/50
    [1m372/373[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 194ms/step - accuracy: 0.9160 - loss: 0.2484
    Epoch 29: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.9145 - loss: 0.2509 - val_accuracy: 0.8461 - val_loss: 0.4614 - learning_rate: 5.0000e-04
    Epoch 30/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.9215 - loss: 0.2184 - val_accuracy: 0.8864 - val_loss: 0.3374 - learning_rate: 2.5000e-04
    Epoch 31/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 194ms/step - accuracy: 0.9284 - loss: 0.2016 - val_accuracy: 0.9066 - val_loss: 0.3027 - learning_rate: 2.5000e-04
    Epoch 32/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m71s[0m 189ms/step - accuracy: 0.9288 - loss: 0.2088 - val_accuracy: 0.8582 - val_loss: 0.4230 - learning_rate: 2.5000e-04
    Epoch 33/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 194ms/step - accuracy: 0.9258 - loss: 0.2047 - val_accuracy: 0.8972 - val_loss: 0.3084 - learning_rate: 2.5000e-04
    Epoch 34/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 197ms/step - accuracy: 0.9282 - loss: 0.2032 - val_accuracy: 0.8696 - val_loss: 0.4039 - learning_rate: 2.5000e-04
    Epoch 35/50
    [1m372/373[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 193ms/step - accuracy: 0.9324 - loss: 0.1946
    Epoch 35: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 195ms/step - accuracy: 0.9293 - loss: 0.1972 - val_accuracy: 0.8710 - val_loss: 0.4054 - learning_rate: 2.5000e-04
    Epoch 36/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 195ms/step - accuracy: 0.9356 - loss: 0.1820 - val_accuracy: 0.9039 - val_loss: 0.2890 - learning_rate: 1.2500e-04
    Epoch 37/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 194ms/step - accuracy: 0.9367 - loss: 0.1794 - val_accuracy: 0.8958 - val_loss: 0.3271 - learning_rate: 1.2500e-04
    Epoch 38/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 198ms/step - accuracy: 0.9306 - loss: 0.1912 - val_accuracy: 0.9207 - val_loss: 0.2667 - learning_rate: 1.2500e-04
    Epoch 39/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 196ms/step - accuracy: 0.9408 - loss: 0.1754 - val_accuracy: 0.9073 - val_loss: 0.2891 - learning_rate: 1.2500e-04
    Epoch 40/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m80s[0m 191ms/step - accuracy: 0.9375 - loss: 0.1769 - val_accuracy: 0.9059 - val_loss: 0.2972 - learning_rate: 1.2500e-04
    Epoch 41/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 193ms/step - accuracy: 0.9396 - loss: 0.1762 - val_accuracy: 0.9093 - val_loss: 0.2920 - learning_rate: 1.2500e-04
    Epoch 42/50
    [1m372/373[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 198ms/step - accuracy: 0.9366 - loss: 0.1774
    Epoch 42: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m75s[0m 200ms/step - accuracy: 0.9400 - loss: 0.1691 - val_accuracy: 0.9099 - val_loss: 0.3015 - learning_rate: 1.2500e-04
    Epoch 43/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m75s[0m 201ms/step - accuracy: 0.9396 - loss: 0.1719 - val_accuracy: 0.9160 - val_loss: 0.2729 - learning_rate: 6.2500e-05
    Epoch 44/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 196ms/step - accuracy: 0.9405 - loss: 0.1675 - val_accuracy: 0.9140 - val_loss: 0.2880 - learning_rate: 6.2500e-05
    Epoch 45/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m75s[0m 201ms/step - accuracy: 0.9397 - loss: 0.1628 - val_accuracy: 0.9241 - val_loss: 0.2560 - learning_rate: 6.2500e-05
    Epoch 46/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m75s[0m 200ms/step - accuracy: 0.9431 - loss: 0.1577 - val_accuracy: 0.9140 - val_loss: 0.2948 - learning_rate: 6.2500e-05
    Epoch 47/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m75s[0m 200ms/step - accuracy: 0.9433 - loss: 0.1602 - val_accuracy: 0.9214 - val_loss: 0.2724 - learning_rate: 6.2500e-05
    Epoch 48/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 197ms/step - accuracy: 0.9466 - loss: 0.1503 - val_accuracy: 0.9214 - val_loss: 0.2713 - learning_rate: 6.2500e-05
    Epoch 49/50
    [1m372/373[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 196ms/step - accuracy: 0.9427 - loss: 0.1575
    Epoch 49: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 199ms/step - accuracy: 0.9446 - loss: 0.1548 - val_accuracy: 0.9207 - val_loss: 0.2675 - learning_rate: 6.2500e-05
    Epoch 50/50
    [1m373/373[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 199ms/step - accuracy: 0.9458 - loss: 0.1564 - val_accuracy: 0.9207 - val_loss: 0.2773 - learning_rate: 3.1250e-05


## Evaluasi dan Visualisasi


```python
train_loss, train_acc = model.evaluate(train_ds, verbose=0)
test_loss,  test_acc  = model.evaluate(test_ds,  verbose=0)
print(f"Akurasi Training : {train_acc*100:.2f}%")
print(f"Akurasi Testing  : {test_acc*100:.2f}%")

if train_acc >= 0.95 and test_acc >= 0.95:
    print("Memenuhi saran: akurasi train & test >= 95%.")
elif train_acc >= 0.85 and test_acc >= 0.85:
    print("Memenuhi kriteria wajib di atas 85%)")
else:
    print("Akurasi masih di bawah 85%")
```

    Akurasi Training : 96.37%
    Akurasi Testing  : 90.76%
    Memenuhi kriteria wajib di atas 85%)



```python
# ====================================
# Plot akurasi & loss selama training
# ====================================
acc      = history.history["accuracy"]
val_acc  = history.history["val_accuracy"]
loss     = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training")
plt.plot(epochs_range, val_acc, label="Validation")
plt.title("Accuracy Model"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training")
plt.plot(epochs_range, val_loss, label="Validation")
plt.title("Loss Model"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](Notebook_DL_files/Notebook_DL_17_0.png)
    



```python
# ========================================
# Confusion matrix & classification report
# ========================================
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.ylabel("Actual Label"); plt.xlabel("Prediction")
plt.tight_layout()
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))
```


    
![png](Notebook_DL_files/Notebook_DL_18_0.png)
    


                  precision    recall  f1-score   support
    
          anjing       0.98      0.87      0.92       487
            ayam       0.88      0.91      0.89       311
       kupu_kupu       0.77      0.95      0.85       212
       laba_laba       0.94      0.93      0.93       483
    
        accuracy                           0.91      1493
       macro avg       0.89      0.91      0.90      1493
    weighted avg       0.92      0.91      0.91      1493
    


## Konversi Model


```python
# ===================================
# Konversi & simpan model ke 3 format
# ===================================

import os
import sys
import shutil
import subprocess
import importlib.util
import tensorflow as tf

from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')

# ---- Konfigurasi ----
SUBMISSION_DIR = "/content/drive/MyDrive/submission"
SAVED_MODEL_DIR = os.path.join(SUBMISSION_DIR, "saved_model")
TFLITE_DIR      = os.path.join(SUBMISSION_DIR, "tflite")
TFJS_DIR        = os.path.join(SUBMISSION_DIR, "tfjs_model")

TFJS_SHARD_SIZE_BYTES = 1024 * 1024 * 1024

def bersihkan_direktori(path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)

def pastikan_dependensi() -> None:
    for paket_pip, nama_modul in [("tensorflowjs", "tensorflowjs"),
                                  ("tf-keras", "tf_keras")]:
        if importlib.util.find_spec(nama_modul) is None:
            print(f"[Setup] Memasang {paket_pip} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", paket_pip],
                           check=True)

def ekspor_saved_model(keras_model, output_dir) -> None:
    output_dir = Path(output_dir)
    bersihkan_direktori(output_dir)
    keras_model.export(str(output_dir))

def ekspor_tflite(saved_model_dir, output_dir, labels: list) -> None:
    saved_model_dir, output_dir = Path(saved_model_dir), Path(output_dir)
    bersihkan_direktori(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    (output_dir / "model.tflite").write_bytes(tflite_model)
    (output_dir / "label.txt").write_text("\n".join(labels), encoding="utf-8")

def ekspor_tfjs(saved_model_dir, output_dir) -> None:
    saved_model_dir, output_dir = Path(saved_model_dir), Path(output_dir)
    bersihkan_direktori(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    perintah = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        f"--weight_shard_size_bytes={TFJS_SHARD_SIZE_BYTES}",
        str(saved_model_dir.resolve()),
        str(output_dir.resolve()),
    ]

    hasil = subprocess.run(perintah, env=env, capture_output=True, text=True)
    if hasil.returncode != 0:
        raise RuntimeError(
            f"--- STDOUT ---\n{hasil.stdout}\n"
            f"--- STDERR ---\n{hasil.stderr}"
        )

def verifikasi_struktur(base_dir) -> None:
    base_dir = Path(base_dir)
    berkas_wajib = [
        Path(SAVED_MODEL_DIR) / "saved_model.pb",
        Path(SAVED_MODEL_DIR) / "variables",
        Path(TFLITE_DIR) / "model.tflite",
        Path(TFLITE_DIR) / "label.txt",
        Path(TFJS_DIR) / "model.json",
    ]
    hilang = [p for p in berkas_wajib if not p.exists()]

    shard_bin = list(Path(TFJS_DIR).glob("*.bin")) if Path(TFJS_DIR).exists() else []
    if not shard_bin:
        hilang.append(Path(TFJS_DIR) / "group1-shard1of1.bin")

    if hilang:
        raise FileNotFoundError(
            "Berkas berikut hilang:\n" + "\n".join(f" - {p}" for p in hilang)
        )

    print("Seluruh berkas wajib berhasil dibuat.\n")
    print("Struktur direktori submission:")
    for root, _, files in os.walk(base_dir):
        level = Path(root).relative_to(base_dir).parts
        print("  " * len(level) + Path(root).name + "/")
        for berkas in sorted(files):
            print("  " * (len(level) + 1) + berkas)

# ---- Orkestrasi ----
if 'model' not in locals() and 'model' not in globals():
    print("Error: Variabel 'model' tidak ditemukan. Pastikan cell training sudah dijalankan.")
else:
    SUBMISSION_DIR = Path(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    pastikan_dependensi()

    ekspor_saved_model(model, SAVED_MODEL_DIR)
    print("SavedModel :", sorted(os.listdir(SAVED_MODEL_DIR)))

    ekspor_tflite(SAVED_MODEL_DIR, TFLITE_DIR, class_names)
    print("TF-Lite :", sorted(os.listdir(TFLITE_DIR)))

    ekspor_tfjs(SAVED_MODEL_DIR, TFJS_DIR)
    print("TF.js :", sorted(os.listdir(TFJS_DIR)))

    print()
    verifikasi_struktur(SUBMISSION_DIR)
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    [Setup] Memasang tensorflowjs ...
    Saved artifact at '/content/drive/MyDrive/submission/saved_model'. The following endpoints are available:
    
    * Endpoint 'serve'
      args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32, name='keras_tensor_5')
    Output Type:
      TensorSpec(shape=(None, 4), dtype=tf.float32, name=None)
    Captures:
      138893116372560: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138893116374864: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138893116373520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138893116374288: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138893116374096: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138893116374480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138893116372752: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984093456: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984095568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984095760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984094224: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984094992: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984093648: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984093264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984095184: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984096912: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984096720: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984095376: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984096528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984095952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984093840: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984098064: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984097872: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984096144: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984097680: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984097104: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984094032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984099216: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984099024: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984097296: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984098832: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984098256: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984096336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984100368: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984100176: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984098448: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138893116373328: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984098640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984100560: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984101328: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984101136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984097488: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984100944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984099792: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984100752: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984102480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984102288: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984099984: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984102096: TensorSpec(shape=(), dtype=tf.resource, name=None)
      138891984101520: TensorSpec(shape=(), dtype=tf.resource, name=None)
    SavedModel : ['assets', 'fingerprint.pb', 'saved_model.pb', 'variables']
    TF-Lite : ['label.txt', 'model.tflite']
    TF.js : ['group1-shard1of1.bin', 'model.json']
    
    Seluruh berkas wajib berhasil dibuat.
    
    Struktur direktori submission:
    submission/
      saved_model/
        fingerprint.pb
        saved_model.pb
        variables/
          variables.data-00000-of-00001
          variables.index
        assets/
      tflite/
        label.txt
        model.tflite
      tfjs_model/
        group1-shard1of1.bin
        model.json


## Inference


```python
# ===================================
# Inferensi menggunakan model TF-Lite
# ===================================

with open(os.path.join(TFLITE_DIR, "label.txt")) as f:
    label_infer = [l.strip() for l in f.readlines() if l.strip()]

# Load interpreter TF-Lite
interpreter = tf.lite.Interpreter(model_path=os.path.join(TFLITE_DIR, "model.tflite"))
interpreter.allocate_tensors()
input_detail  = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

def prediksi_tflite(citra):
    x = tf.expand_dims(citra, 0)
    x = tf.cast(x, input_detail["dtype"]).numpy()
    interpreter.set_tensor(input_detail["index"], x)
    interpreter.invoke()
    keluaran = interpreter.get_tensor(output_detail["index"])[0]
    return label_infer[int(np.argmax(keluaran))], float(np.max(keluaran))

# Ambil beberapa gambar dari test set
bukti_ds = muat_dataset("test", acak=True).take(1)
plt.figure(figsize=(14, 7))
for images, labels in bukti_ds:
    for i in range(min(8, images.shape[0])):
        pred_label, conf = prediksi_tflite(images[i])
        asli = class_names[int(labels[i])]
        ax = plt.subplot(2, 4, i + 1)
        ax.imshow(images[i].numpy().astype("uint8"))
        warna = "green" if pred_label == asli else "red"
        ax.set_title(f"Prediksi: {pred_label} ({conf:.0%})\nAsli: {asli}",
                     color=warna, fontsize=9)
        ax.axis("off")
    break
plt.tight_layout()
plt.show()
```

    Found 1493 files belonging to 4 classes.


    /usr/local/lib/python3.12/dist-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
        TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
        See the [migration guide](https://ai.google.dev/edge/litert/migration)
        for details.
        
      warnings.warn(_INTERPRETER_DELETION_WARNING)



    
![png](Notebook_DL_files/Notebook_DL_22_2.png)
    


## Save requirements.txt


```python
# ========================
# Membuat requirements.txt
# ========================
!pip freeze > requirements.txt
print("requirements.txt berhasil dibuat.")
```

    requirements.txt berhasil dibuat.



```python
!jupyter nbconvert --to markdown Notebook_DL.ipynb
```

    [NbConvertApp] WARNING | pattern 'Notebook_DL.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

