# Deep Learning_1

An end-to-end image-classification project built with TensorFlow/Keras: a custom convolutional neural network (CNN) trained from scratch to classify photographs of four animal categories, then exported to three deployment formats (TensorFlow SavedModel, TensorFlow Lite, and TensorFlow.js).

| | |
|---|---|
| **Internal project title** | *Proyek Klasifikasi Gambar: Klasifikasi Hewan* (Image Classification Project: Animal Classification) |
| **Task** | Supervised multi-class image classification (4 classes) |
| **Framework** | TensorFlow 2.20.0 / Keras 3.13.2 |
| **Entry point** | `notebook.ipynb.ipynb` |
| **Primary language of code comments & narrative** | Indonesian |
| **Deliverable type** | Course submission (Dicoding-style), with mandatory accuracy criteria checked in-notebook |

### Purpose and learning objectives

The notebook is organised as a complete, auditable deep-learning workflow and demonstrates the following competencies:

1. **Dataset acquisition and curation** — programmatic download, integrity verification, class subsetting, and label translation.
2. **Exploratory verification against explicit acceptance criteria** — the notebook asserts a minimum dataset size (≥ 10,000 images), a minimum number of classes (≥ 3), and heterogeneous native resolutions.
3. **Reproducible preprocessing** — deterministic stratified-by-folder splitting, in-graph rescaling, and train-only data augmentation.
4. **CNN design and training from scratch** — no transfer learning or pretrained backbone is used.
5. **Training control via callbacks** — early stopping, adaptive learning-rate reduction, checkpointing, and a custom accuracy-threshold stopping callback.
6. **Quantitative evaluation** — hold-out test accuracy, confusion matrix, and a per-class precision/recall/F1 report.
7. **Multi-target deployment export** — SavedModel, TFLite (with optimisation enabled), and TF.js graph-model, followed by a structural verification routine.
8. **On-device inference validation** — predictions drawn from the exported TFLite interpreter rather than the in-memory Keras model.

### Intended audience

Machine-learning students and practitioners who need a worked, reviewable reference for the full "data → trained CNN → multi-format deployment artifact" pipeline; and reviewers/graders who need to confirm that the stated criteria were met.

---

## Table of contents

- [Deep Learning\_1](#deep-learning_1)
    - [Purpose and learning objectives](#purpose-and-learning-objectives)
    - [Intended audience](#intended-audience)
  - [Table of contents](#table-of-contents)
  - [📁 Project contents](#-project-contents)
  - [Workflow](#workflow)
  - [Dataset](#dataset)
  - [Model architecture](#model-architecture)
  - [Training configuration](#training-configuration)
  - [Recorded results](#recorded-results)
    - [Headline metrics](#headline-metrics)
    - [Per-class test performance (n = 1,493)](#per-class-test-performance-n--1493)
    - [Caveat on the training-accuracy figure](#caveat-on-the-training-accuracy-figure)
    - [Figures saved in the notebook](#figures-saved-in-the-notebook)
  - [Exported model artifacts](#exported-model-artifacts)
    - [Verified signatures](#verified-signatures)
    - [Quantisation note (TFLite)](#quantisation-note-tflite)
    - [Size summary](#size-summary)
  - [Environment and dependencies](#environment-and-dependencies)
    - [Recorded runtime](#recorded-runtime)
    - [Dependencies installed by the notebook at runtime but absent from `requirements.txt`](#dependencies-installed-by-the-notebook-at-runtime-but-absent-from-requirementstxt)
    - [Minimal dependency set for re-running the notebook](#minimal-dependency-set-for-re-running-the-notebook)
  - [Running the project](#running-the-project)
    - [Prerequisites you must supply](#prerequisites-you-must-supply)
    - [Recommended execution path (Google Colab)](#recommended-execution-path-google-colab)
    - [Running outside Colab](#running-outside-colab)
    - [Expected runtime outputs](#expected-runtime-outputs)
  - [Using the exported models](#using-the-exported-models)
    - [TensorFlow Lite](#tensorflow-lite)
    - [TensorFlow SavedModel](#tensorflow-savedmodel)
    - [TensorFlow.js](#tensorflowjs)
  - [Reproducibility barriers and known issues](#reproducibility-barriers-and-known-issues)
    - [Blocking / high impact](#blocking--high-impact)
    - [Documentation and consistency conflicts](#documentation-and-consistency-conflicts)
    - [Methodological / design observations](#methodological--design-observations)
  - [Verification status of this document](#verification-status-of-this-document)
  - [Not specified in the provided files](#not-specified-in-the-provided-files)
  - [Suggested improvements](#suggested-improvements)
  - [Attribution and licence](#attribution-and-licence)

---

## 📁 Project contents

Filenames below are reproduced exactly as they appear in the archive, including the doubled extension on the notebook.

| Path | Purpose | Role in the workflow |
|---|---|---|
| `notebook.ipynb.ipynb` | The complete, executed project notebook (26 cells: 16 code, 10 Markdown) with all outputs retained, including rendered figures. | **Sole entry point and sole source of executable logic.** Contains every stage: dependency install, data download, curation, EDA, split, augmentation, model definition, training, evaluation, export, and TFLite inference. |
| `requirements.txt` | Output of `pip freeze` from the execution runtime — 703 pinned entries describing the entire environment, not a curated dependency list. | Environment record. **Not directly installable as-is** (see [Reproducibility barriers](#reproducibility-barriers-and-known-issues)). |
| `saved_model/saved_model.pb` | Serialised TensorFlow graph and signature definitions. | Canonical full-precision export; **also the source artifact** from which the TFLite and TF.js versions are converted. |
| `saved_model/variables/variables.data-00000-of-00001` | Trained weight values (5,236,341 bytes). | Weight payload for the SavedModel. |
| `saved_model/variables/variables.index` | Index mapping variable names to offsets in the data shard. | Required companion to the weight shard. |
| `saved_model/fingerprint.pb` | TensorFlow SavedModel fingerprint (99 bytes). | Integrity/identity metadata emitted automatically on export. |
| `saved_model/assets/` | Empty directory (present in the archive, contains no files). | Standard SavedModel layout element; this model requires no external asset files. |
| `tflite/model.tflite` | TensorFlow Lite FlatBuffer (682,088 bytes), converted from the SavedModel with `tf.lite.Optimize.DEFAULT`. | Mobile/edge deployment artifact; **also the model used for the notebook's final inference demonstration**. |
| `tflite/label.txt` | Four newline-separated class labels, in model output-index order: `anjing`, `ayam`, `kupu_kupu`, `laba_laba`. | Maps the model's 4-unit softmax output to human-readable class names. Written from `class_names` at export time, so index order is guaranteed consistent with training. |
| `tfjs_model/model.json` | TF.js **graph-model** topology, signature, and weights manifest (37 weight tensors: 36 × `float32`, 1 × `int32`). Converted by TensorFlow.js Converter v4.22.0. | Browser/Node.js deployment artifact. |
| `tfjs_model/group1-shard1of1.bin` | Single weight shard (2,603,680 bytes) — the conversion used a 1 GiB shard-size limit, so all weights fit in one file. | Weight payload for the TF.js model; must sit alongside `model.json`. |

**Not present in the archive:** no `.py` modules, no configuration files (`.yaml` / `.json` / `.ini` / `.env`), no dataset files or images, no pre-existing `README`, no `LICENSE`, no test suite, no CI configuration, and none of the intermediate files the notebook creates at runtime (`/content/model_terbaik.keras`, the working/split dataset directories, or the `Notebook_DL.md` Markdown export produced by the final cell).

---

## Workflow

The notebook executes the following stages in order. Later cells depend on in-memory state from earlier cells (notably `model`, `class_names`, `history`, `muat_dataset`, and the export-path constants), so **the notebook must be run top to bottom**.

| # | Notebook section | Key operations |
|---|---|---|
| 1 | *Import Semua Library* | Installs `split-folders`; imports TensorFlow/Keras, scikit-learn metrics, Pillow, NumPy, pandas, Matplotlib; seeds `random`, `numpy`, and `tf.random` with `SEED = 42`; prints TF/Keras versions and GPU availability. |
| 2 | *Data Preparation* | Downloads the Kaggle dataset via `kagglehub`; locates the `raw-img` class-parent directory by recursive glob. |
| 3 | — | Subsets 4 of 10 classes, translates Italian folder names to Indonesian, validates every file with `PIL.Image.verify()`, and copies survivors into a flat working directory. |
| 4 | — | Summarises per-class counts and the set of unique native resolutions; **asserts** the acceptance criteria; plots the class-distribution bar chart. |
| 5 | — | Renders a 4-sample-per-class image grid with native resolutions in the titles. |
| 6 | *Data Preprocessing → Split Dataset* | `splitfolders.ratio(..., seed=42, ratio=(0.8, 0.1, 0.1))`; builds three `tf.data` pipelines at 160×160 / batch 32; applies augmentation to **train only** via `.map`; caches and prefetches val/test. |
| 7 | *Modelling* | Defines and compiles the sequential CNN; prints the layer summary. |
| 8 | — | Defines the custom `StopPada95` callback; assembles the callback list; runs `model.fit` for up to 50 epochs. |
| 9 | *Evaluasi dan Visualisasi* | Evaluates on train and test; prints a pass/fail message against the 95% / 85% thresholds; plots accuracy and loss curves; computes the confusion matrix and `classification_report`. |
| 10 | *Konversi Model* | Mounts Google Drive; ensures `tensorflowjs` and `tf-keras` are installed; exports SavedModel → TFLite → TF.js; runs `verifikasi_struktur()` to assert every required output file exists and prints the resulting directory tree. |
| 11 | *Inference* | Loads `label.txt`, instantiates `tf.lite.Interpreter` on `model.tflite`, and displays 8 test images annotated with predicted label, confidence, and ground truth (green/red title colour by correctness). |
| 12 | *Save requirements.txt* | `pip freeze > requirements.txt`, then an `nbconvert` call to produce a Markdown rendering of the notebook. |

---

## Dataset

**Source:** the public Kaggle dataset `alessiocorrado99/animals10`, fetched at runtime with `kagglehub.dataset_download(...)`. The images themselves are **not bundled in this archive** and must be re-downloaded.

The source dataset contains ten classes (`cane`, `cavallo`, `elefante`, `farfalla`, `gallina`, `gatto`, `mucca`, `pecora`, `ragno`, `scoiattolo`). The notebook deliberately retains four, and renames them:

| Source folder (Italian) | Project label (Indonesian) | English | Images |
|---|---|---|---|
| `cane` | `anjing` | dog | 4,863 |
| `gallina` | `ayam` | chicken | 3,098 |
| `farfalla` | `kupu_kupu` | butterfly | 2,112 |
| `ragno` | `laba_laba` | spider | 4,821 |

**Curated totals (as recorded in the notebook outputs):**

- 14,894 valid images copied; 0 files skipped as corrupt or non-image.
- 724 distinct native resolutions (e.g. 449×640, 640×320, 300×270, 181×300) — i.e. the input is genuinely heterogeneous and is normalised only by the loader's resize to 160×160.
- Class imbalance ratio ≈ **2.30 : 1** (largest `anjing` 4,863 vs smallest `kupu_kupu` 2,112). No class weighting or resampling is applied anywhere in the notebook.

**Splits** (`splitfolders`, seed 42, 80/10/10):

| Split | Images |
|---|---|
| train | 11,913 |
| validation | 1,488 |
| test | 1,493 |

---

## Model architecture

A custom `Sequential` CNN named `cnn_klasifikasi_hewan`, trained from scratch — **no transfer learning, no pretrained weights**.

```
Input (160, 160, 3)
Rescaling(1/255)                              # normalisation lives INSIDE the graph

Conv2D(32, 3, same, relu) → BatchNorm
Conv2D(32, 3, same, relu) → BatchNorm → MaxPool → Dropout(0.25)

Conv2D(64, 3, same, relu) → BatchNorm
Conv2D(64, 3, same, relu) → BatchNorm → MaxPool → Dropout(0.25)

Conv2D(128, 3, same, relu) → BatchNorm
Conv2D(128, 3, same, relu) → BatchNorm → MaxPool → Dropout(0.30)

Conv2D(256, 3, same, relu) → BatchNorm → MaxPool → Dropout(0.30)

GlobalAveragePooling2D
Dense(256, relu) → BatchNorm → Dropout(0.50)
Dense(4, softmax)
```

| Parameter count | Value |
|---|---|
| Total | 652,836 (2.49 MB) |
| Trainable | 650,916 (2.48 MB) |
| Non-trainable (BatchNorm statistics) | 1,920 (7.50 KB) |

**Design note worth preserving in any refactor:** placing `Rescaling(1/255)` inside the model means all three exported artifacts accept raw `[0, 255]` pixel values directly. Callers must **not** pre-divide inputs, or the model will be fed values in `[0, 1/255]`.

---

## Training configuration

| Setting | Value |
|---|---|
| Input size | 160 × 160 × 3 |
| Batch size | 32 |
| Optimiser | Adam, initial learning rate `1e-3` |
| Loss | `sparse_categorical_crossentropy` |
| Metric | `accuracy` |
| Max epochs | 50 |
| Random seed | 42 (`random`, `numpy`, `tf.random`, and the split) |
| Augmentation (train only) | `RandomFlip("horizontal")`, `RandomRotation(0.15)`, `RandomZoom(0.15)`, `RandomContrast(0.10)` |

**Callbacks**

| Callback | Configuration | Behaviour in the recorded run |
|---|---|---|
| `EarlyStopping` | `monitor="val_accuracy"`, `patience=10`, `restore_best_weights=True` | Never triggered — training ran the full 50 epochs. |
| `ReduceLROnPlateau` | `monitor="val_loss"`, `factor=0.5`, `patience=4`, `min_lr=1e-6` | Fired at epochs 22, 29, 35, 42, 49; learning rate decayed `1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → 3.125e-5`. |
| `ModelCheckpoint` | `"/content/model_terbaik.keras"`, `monitor="val_accuracy"`, `save_best_only=True` | Wrote a best-weights checkpoint to the Colab local disk. **This file is not in the archive.** |
| `StopPada95` (custom) | Stops when train **and** validation accuracy both reach 0.95 | Never triggered. |

**Recorded hardware:** Google Colab with a GPU accelerator; the notebook metadata records `gpuType: "T4"`. Epoch wall time was roughly 71–113 s, so the full 50-epoch run took on the order of **one hour**.

---

## Recorded results

> These figures are **saved outputs read from the notebook**. They are the author's recorded results. They were **not** re-run or independently reproduced while writing this document.

### Headline metrics

| Metric | Value |
|---|---|
| Training accuracy (`model.evaluate(train_ds)`) | 96.37% |
| **Test accuracy (`model.evaluate(test_ds)`)** | **90.76%** |
| Best validation accuracy | 92.41% (epoch 45) |
| Final-epoch validation accuracy | 92.07% (epoch 50, `val_loss` 0.2773) |
| Macro-average F1 (test) | 0.90 |
| Weighted-average F1 (test) | 0.91 |

The notebook's own criterion check printed: *"Memenuhi kriteria wajib di atas 85%"* — the mandatory >85% threshold is met on both train and test; the optional ≥95% train-and-test target is **not** met (test = 90.76%).

### Per-class test performance (n = 1,493)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `anjing` | 0.98 | 0.87 | 0.92 | 487 |
| `ayam` | 0.88 | 0.91 | 0.89 | 311 |
| `kupu_kupu` | **0.77** | 0.95 | 0.85 | 212 |
| `laba_laba` | 0.94 | 0.93 | 0.93 | 483 |
| **accuracy** | | | **0.91** | 1,493 |
| macro avg | 0.89 | 0.91 | 0.90 | 1,493 |
| weighted avg | 0.92 | 0.91 | 0.91 | 1,493 |

**Reading the error structure:** the smallest class, `kupu_kupu`, has high recall (0.95) but the lowest precision (0.77) — it is systematically *over-predicted*, absorbing false positives from the two large classes. Conversely `anjing` has near-perfect precision (0.98) but the weakest recall (0.87). This is the signature of unmitigated class imbalance (2.30 : 1) rather than insufficient model capacity.

### Caveat on the training-accuracy figure

Augmentation is applied to `train_ds` with `training=True` inside a `.map()` call, so the pipeline still augments when `model.evaluate(train_ds)` is invoked. The reported 96.37% is therefore accuracy **on augmented training data**, which understates the clean training fit. The true train/test generalisation gap is wider than the apparent 5.6 pp.

### Figures saved in the notebook

Four rendered figures are embedded in the notebook outputs (PNG): the class-distribution bar chart, the sample-image grid, the paired accuracy/loss learning curves, and the test-set confusion matrix. The confusion-matrix cell plots raw counts; the numeric matrix itself is only available as the rendered image, not as serialised data.

---

## Exported model artifacts

All three exports derive from the **same trained weights**. The SavedModel is produced first via `model.export()`, and both other formats are converted from it.

### Verified signatures

The following were read directly from the binary files during this review:

| Artifact | Signature key | Input tensor | Output tensor | Shape |
|---|---|---|---|---|
| SavedModel | `serving_default` | `keras_tensor_5` (`serving_default_keras_tensor_5:0`) | `Identity` | in `(None, 160, 160, 3)` float32 → out `(None, 4)` float32 |
| TFLite | `serving_default` | `keras_tensor_5` | `output_0` | same logical shape; batch dimension fixed by the interpreter at allocation |
| TF.js | — (graph-model) | `keras_tensor_5:0` | `Identity:0` / `output_0` | in `(-1, 160, 160, 3)` `DT_FLOAT` → out `(-1, 4)` `DT_FLOAT` |

The SavedModel's embedded producer version string is `2.20.0`, consistent with the `tensorflow==2.20.0` pin in `requirements.txt`.

### Quantisation note (TFLite)

The converter is configured with `converter.optimizations = [tf.lite.Optimize.DEFAULT]`, which applies **dynamic-range weight quantisation**. This is confirmed by inspection: the FlatBuffer contains `tfl.pseudo_qconst` constant tensors, and the file is 682 KB against ~2.49 MB of float32 parameters — roughly a 4× reduction, as expected for int8 weights with float activations.

**Consequence:** `model.tflite` is numerically **not** identical to the SavedModel. The notebook does not re-measure accuracy after quantisation, so no quantised-model accuracy figure exists in the archive. The 90.76% test accuracy applies to the full-precision Keras model only.

### Size summary

| Artifact | On-disk size |
|---|---|
| SavedModel (graph + weights + fingerprint) | ≈ 5.59 MB |
| TF.js (`model.json` + 1 shard) | ≈ 2.63 MB |
| TFLite (quantised) | ≈ 0.68 MB |

---

## Environment and dependencies

### Recorded runtime

| Component | Version (as recorded) |
|---|---|
| TensorFlow | 2.20.0 |
| Keras | 3.13.2 |
| NumPy | 2.0.2 |
| pandas | 2.2.2 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.10.0 |
| Pillow | 11.3.0 |
| kagglehub | 1.0.2 |
| tf-keras | 2.20.0 |
| TensorFlow.js Converter | 4.22.0 (from `tfjs_model/model.json`) |
| Platform | Google Colab, GPU (T4) |

**Python version is reported inconsistently across the archive** — see [Reproducibility barriers](#reproducibility-barriers-and-known-issues).

### Dependencies installed by the notebook at runtime but absent from `requirements.txt`

`requirements.txt` was generated by the cell with execution count **1** — i.e. it was captured *before* the export cell ran in that session — so it does not record everything the notebook needs:

| Package | Installed in | Present in `requirements.txt`? |
|---|---|---|
| `split-folders` (imported as `splitfolders`) | First code cell (`!pip install split-folders --quiet`) | ❌ **Missing** |
| `tensorflowjs` | Export cell, installed on demand by the `pastikan_dependensi()` guard (`subprocess` + pip) | ❌ **Missing** |
| `tf-keras` | Same guard (already present in the Colab base image, so not re-installed) | ✅ `tf_keras==2.20.0` |

Both missing packages must be installed manually for a clean re-run.

### Minimal dependency set for re-running the notebook

Derived from the notebook's actual imports and runtime installs. Exact pins are those recorded above; `split-folders` and `tensorflowjs` versions are **not specified in the provided files**, so no version is asserted here.

```
tensorflow==2.20.0
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
matplotlib==3.10.0
pillow==11.3.0
kagglehub==1.0.2
tf-keras==2.20.0
split-folders        # version not specified in the provided files
tensorflowjs         # version not specified in the provided files
```

---

## Running the project

### Prerequisites you must supply

1. **A Kaggle account and API credentials.** `kagglehub.dataset_download("alessiocorrado99/animals10")` requires authentication and will prompt for, or read, Kaggle credentials. **No credentials are included in this archive, and none should be committed to it.** Configure them in your own environment (the standard `~/.kaggle/kaggle.json` or the equivalent Kaggle/kagglehub environment variables).
2. **~2–3 GB of free disk** for the downloaded dataset plus the curated copy plus the 80/10/10 split (the notebook copies files twice rather than symlinking).
3. **A GPU.** The recorded run took ~75 s/epoch on a T4 for 50 epochs. CPU-only training is possible but substantially slower.
4. **A Google Drive account** *if* you run the export cell unmodified — it calls `google.colab.drive.mount()`. See the path edits below.

### Recommended execution path (Google Colab)

The notebook was authored for and executed in Colab, and is the lowest-friction option because `/content/...` paths and the Drive mount work as written.

1. Upload `notebook.ipynb.ipynb` to Colab.
2. Set the runtime to a GPU accelerator.
3. Run all cells top to bottom.

### Running outside Colab

Three code changes are required. The notebook has no configuration layer, so these must be edited in the cells themselves:

| What to change | Where | Current hard-coded value |
|---|---|---|
| Working/curated dataset directory | Data-curation cell | `WORK_DIR = "/content/dataset_kerja"` |
| Split output directory | Split cell | `OUTPUT_SPLIT = "/content/dataset_split"` |
| Best-checkpoint path | `ModelCheckpoint` in the callbacks cell | `"/content/model_terbaik.keras"` |
| Export root **and** the Drive mount | Export cell | `SUBMISSION_DIR = "/content/drive/MyDrive/submission"`; remove or guard the `from google.colab import drive` / `drive.mount(...)` lines, which will fail outside Colab |

Also note that the final cell runs `!jupyter nbconvert --to markdown Notebook_DL.ipynb`, which references a filename that does not exist in this archive; update it to the actual notebook filename or skip that cell.

### Expected runtime outputs

Running the notebook end to end recreates the directory structure shipped in this archive:

```
<SUBMISSION_DIR>/
├── saved_model/
│   ├── assets/
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── tflite/
│   ├── label.txt
│   └── model.tflite
└── tfjs_model/
    ├── group1-shard1of1.bin
    └── model.json
```

The `verifikasi_struktur()` routine raises `FileNotFoundError` if any required file is missing, so a successful export is self-checking.

---

## Using the exported models

> ⚠️ The snippets below are **illustrative documentation**, adapted from the notebook's own inference cell and from the standard loading APIs for each format. They were **not executed or verified** during the preparation of this README — TensorFlow is not installed in the environment where this document was written, and the task scope excluded installing packages. Treat them as a starting point to validate in your own environment.

**Input contract, common to all three artifacts:** a float tensor of shape `(batch, 160, 160, 3)` containing **raw pixel values in `[0, 255]`**. Rescaling is inside the graph — do not normalise beforehand. The output is a `(batch, 4)` softmax vector whose indices map to the lines of `tflite/label.txt` in order: `0 → anjing`, `1 → ayam`, `2 → kupu_kupu`, `3 → laba_laba`.

### TensorFlow Lite

Adapted directly from the notebook's inference cell:

```python
import numpy as np
import tensorflow as tf

with open("tflite/label.txt") as f:
    labels = [line.strip() for line in f if line.strip()]

interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

# image: a (160, 160, 3) array of raw pixel values in [0, 255]
x = np.expand_dims(image, 0).astype(input_detail["dtype"])
interpreter.set_tensor(input_detail["index"], x)
interpreter.invoke()
probs = interpreter.get_tensor(output_detail["index"])[0]

print(labels[int(np.argmax(probs))], float(np.max(probs)))
```

Note: the recorded run emitted a deprecation warning — `tf.lite.Interpreter` is scheduled for removal, with migration directed to the LiteRT interpreter in the `ai-edge-litert` package. The archive does not include LiteRT-based code.

### TensorFlow SavedModel

The export was produced with `model.export()`, which writes an inference-only SavedModel exposing the `serving_default` signature (input `keras_tensor_5`, output `Identity`). Load it as a serving artifact rather than expecting a reloaded trainable Keras object:

```python
import tensorflow as tf

loaded = tf.saved_model.load("saved_model")
serve = loaded.signatures["serving_default"]
out = serve(keras_tensor_5=tf.constant(batch, dtype=tf.float32))   # batch: (N, 160, 160, 3)
```

### TensorFlow.js

`model.json` declares `"format": "graph-model"`, so it must be loaded with the graph-model loader, not `loadLayersModel`. `group1-shard1of1.bin` must be served from the same location as `model.json`.

```javascript
const model = await tf.loadGraphModel('tfjs_model/model.json');
const probs = model.predict(input);   // input: [N, 160, 160, 3], raw pixel values
```

---

## Reproducibility barriers and known issues

These are factual observations from inspecting the archive, ordered roughly by impact.

### Blocking / high impact

1. **`requirements.txt` is not installable as-is.** It is a raw `pip freeze` of a Colab image (703 entries) and contains environment-specific entries that will fail on a generic machine, including a local-file requirement (`google-colab @ file:///colabtools/dist/google_colab-1.0.0.tar.gz`), direct NVIDIA index URLs for RAPIDS/cuCIM/cuVS wheels, a GitHub VCS requirement for `jupyter_kernel_gateway`, system-bound packages (`python-apt==0.0.0`, `dbus-python`, `PyGObject`, `GDAL==3.8.4`), and a spaCy model wheel URL. Use the [minimal dependency set](#minimal-dependency-set-for-re-running-the-notebook) instead, and treat `requirements.txt` as a provenance record.
2. **Two runtime dependencies are missing from `requirements.txt`:** `split-folders` and `tensorflowjs`. A `pip install -r requirements.txt` followed by a notebook run will fail at the `import splitfolders` line.
3. **The dataset is not bundled and requires Kaggle authentication.** Re-running depends on the continued availability of the external Kaggle dataset and on credentials you supply yourself.
4. **Hard-coded absolute paths throughout.** All I/O paths are Colab-specific (`/content/...`) or Drive-specific (`/content/drive/MyDrive/submission`), and the export cell unconditionally calls `google.colab.drive.mount()`. There is no configuration file, CLI argument, or environment-variable layer; paths must be edited in-cell.

### Documentation and consistency conflicts

5. **Python version is reported three different ways.** The notebook metadata records `language_info.version: "3.13.13"`; a warning traceback in the recorded output resolves to `/usr/local/lib/python3.12/dist-packages/...`; and several pinned wheels in `requirements.txt` are `cp312` builds. These cannot all be correct. The `cp312` wheels and the traceback path are the stronger evidence, but the discrepancy is unresolved from the archive alone, so **the exact interpreter version is effectively not specified**.
6. **Filename mismatch.** The archived notebook is `notebook.ipynb.ipynb` (doubled extension), while the final cell runs `nbconvert` against `Notebook_DL.ipynb`. Neither `Notebook_DL.ipynb`, `Notebook_DL.md`, nor the `Notebook_DL_files/` directory referenced in that cell's output is present.
7. **Non-monotonic execution counts.** Cell execution counts run 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25 — and then the last two cells show counts **1** and **15**. The notebook as archived was therefore **not** produced by a single clean top-to-bottom run; at minimum the `requirements.txt` cell was executed in a different (later) session. The saved outputs are consistent with each other for the training/evaluation chain, but the archive is not a verified single-pass artifact.

### Methodological / design observations

8. **Training accuracy is measured on augmented data** (see the [caveat above](#caveat-on-the-training-accuracy-figure)), so the reported 96.37% is not a clean training-fit measure.
9. **Which weights the exports carry is ambiguous.** `EarlyStopping(restore_best_weights=True)` never triggered because training completed all 50 epochs; whether best weights (epoch 45, `val_accuracy` 0.9241) are restored at normal completion depends on Keras-version behaviour. The `ModelCheckpoint` best-weights file is not included, and the exports were created from the in-memory `model` object. From the archive alone it cannot be determined whether the shipped artifacts hold epoch-50 or epoch-45 weights. The recorded test accuracy (90.76%) is whatever the in-memory model scored at that moment.
10. **No post-quantisation evaluation.** TFLite accuracy after `Optimize.DEFAULT` is never measured, so the deployment artifact's accuracy is unknown.
11. **Class imbalance is unaddressed.** No class weights, oversampling, or stratification beyond per-folder ratio splitting. This is the most likely cause of the `kupu_kupu` precision deficit.
12. **The validation split is used for both early-stopping/LR decisions and as the model-selection signal**, which is standard practice here, but note the test split is touched only at evaluation — correctly, no leakage was observed in the code.
13. **No seed control for the augmentation layers.** The `RandomFlip`/`RandomRotation`/`RandomZoom`/`RandomContrast` layers are constructed without explicit `seed=` arguments, so exact run-to-run reproduction of training is not guaranteed even with the global seeds set.
14. **Deprecated inference API.** `tf.lite.Interpreter` is flagged for deletion in favour of LiteRT (`ai-edge-litert`); the notebook's inference cell will need migration.

---

## Verification status of this document

In the interest of accuracy, here is exactly what was and was not done while writing this README.

**Performed:**
- Full extraction and listing of the archive, including empty directories.
- Read every Markdown and code cell of `notebook.ipynb.ipynb` in full.
- Read all recorded textual outputs (stdout/stderr, execution counts, the model summary, the complete 50-epoch training log, the evaluation prints, and the classification report).
- Read `requirements.txt` in full (703 lines) and grepped it for the notebook's imports.
- Parsed `tfjs_model/model.json` — format, converter version, signature, shapes, weight manifest, dtype census.
- Read `tflite/label.txt` in full.
- Extracted and inspected metadata strings from `saved_model/saved_model.pb` and `tflite/model.tflite` to confirm signature keys, tensor names, the TF producer version, and the presence of quantised constants.
- Recorded exact file sizes for every archive member.

**Not performed:**
- No training, fine-tuning, or evaluation was run.
- No inference was executed against any of the three artifacts.
- No packages were installed; TensorFlow is **not** available in the review environment, so neither the SavedModel, the TFLite FlatBuffer, nor the TF.js model was loaded by its native runtime.
- The confusion-matrix figure was not decoded from PNG, so per-cell counts are not reported here.
- No source file in the archive was modified.

**Therefore:** every performance number in this README is a *recorded* result transcribed from the notebook's saved outputs. None of it constitutes independent reproduction or verification of the model's behaviour.

---

## Not specified in the provided files

The following are absent from the archive. Each must be supplied by you (or by the project owner) before the project can be treated as complete or redistributable.

| Item | What you must supply |
|---|---|
| **Licence** | No `LICENSE` file and no licence statement. The terms for reusing the code and the trained weights are undefined. Add an explicit licence. Note separately that the **Kaggle `animals10` dataset carries its own terms**, which are not reproduced here and must be checked at source before any redistribution of derived weights. |
| **Repository URL / version control** | No git metadata, remote, or project URL. |
| **Version of `split-folders` and `tensorflowjs`** | Not captured in `requirements.txt`; pin them yourself after a successful run. |
| **Exact Python interpreter version** | Conflicting evidence (see barrier #5). Confirm against your target environment. |
| **Kaggle credentials** | Not included, and must not be. Configure in your own environment. |
| **Model card / intended-use statement** | No documentation of intended deployment context, failure modes, or out-of-distribution behaviour. |
| **Numeric confusion matrix** | Available only as a rendered image; re-run the evaluation cell if you need the counts as data. |
| **Quantised-model accuracy** | Never measured; evaluate `model.tflite` on the test split to obtain it. |
| **Random-state capture for augmentation** | No per-layer seeds; exact training reproduction is not guaranteed. |
| **Tests / CI** | None present. |

---

## Suggested improvements

**Not implemented in this project** — these are recommendations, listed separately from the archive's actual contents and ordered by expected return on effort.

1. **Address class imbalance.** Pass `class_weight` to `model.fit` (inverse-frequency weights) or oversample `kupu_kupu`. Given that the smallest class carries both the highest recall and the lowest precision, this is the single most likely route to closing the gap to the 95% target.
2. **Measure the quantised model.** Add an evaluation pass over the test split through the TFLite interpreter, and record the accuracy delta against the float model. Without this, the deployment artifact's quality is unknown.
3. **Evaluate training fit on a non-augmented pipeline.** Build a separate, un-augmented `train_eval_ds` for `model.evaluate` so the reported training accuracy is interpretable and the generalisation gap is measured honestly.
4. **Add transfer learning as a baseline comparison.** A frozen pretrained backbone (e.g. an ImageNet-pretrained feature extractor with a new head) would almost certainly exceed 90.76% on four well-separated classes at a fraction of the 50-epoch cost, and would make the from-scratch result interpretable as a deliberate design choice rather than the only option tried.
5. **Externalise configuration.** Lift `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS`, `SEED`, the class subset, and all paths into a single configuration cell or a YAML file, and derive paths from a root variable so the notebook runs unmodified outside Colab.
6. **Serialise the training history and metrics.** Write `history.history`, the confusion matrix, and the classification report to JSON/CSV alongside the model so results are machine-readable and diffable across runs, rather than recoverable only from rendered figures.
7. **Curate a real `requirements.txt`.** Replace the `pip freeze` dump with the minimal direct-dependency set, pinned, and regenerate it *after* the export cell so nothing is missed.
8. **Seed the augmentation layers** and record the resolved seed, to make runs reproducible end to end.
9. **Migrate inference to LiteRT** (`ai-edge-litert`) ahead of the `tf.lite.Interpreter` removal.
10. **Ship the best checkpoint, or export deterministically.** Either include the `ModelCheckpoint` artifact or explicitly reload the best checkpoint before exporting, so there is no ambiguity about which weights the deployment artifacts contain.
11. **Normalise the filenames.** Rename `notebook.ipynb.ipynb` to a single extension and fix the `nbconvert` reference, then re-execute once top to bottom so the archived execution counts reflect a clean run.
12. **Add per-class error analysis.** Surface the highest-confidence misclassifications per class; with 724 distinct source resolutions, it would also be worth checking whether aspect-ratio distortion from the 160×160 resize correlates with error.

---

## Attribution and licence

**Author (as documented in the notebook's own header):** Hartanto Dewantoro 

**Third-party data:** the trained weights derive from the Kaggle dataset `alessiocorrado99/animals10`, which is governed by its own terms at source. Those terms are not reproduced in this archive and should be reviewed before redistributing the model artifacts.
