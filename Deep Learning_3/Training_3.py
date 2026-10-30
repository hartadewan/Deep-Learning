# # 1. Setup, Dependencies, dan Konfigurasi
# %%
import os, sys, json, math, time, random, shutil, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

try:
    import keras
    from keras import layers
except ImportError:                     
    from tensorflow import keras
    from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------- plotting style
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110
plt.rcParams["figure.figsize"] = (12, 4)

print(f"Python      : {sys.version.split()[0]}")
print(f"TensorFlow  : {tf.__version__}")
print(f"Keras       : {keras.__version__}")
print(f"NumPy       : {np.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"statsmodels : {sm.__version__}")

gpus = tf.config.list_physical_devices("GPU")
print(f"\nGPU tersedia: {gpus if gpus else 'TIDAK ADA (training akan berjalan di CPU dan jauh lebih lambat)'}")
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

# ### 1.1 Configuration

# %%
# ======== Global Configuration =========
STUDENT_NAME   = "Hartanto Dewantoro"     

# --- Problem definition ---
HORIZON        = 24                
TARGET         = "Close"           
FREQ           = "h"               

# --- Windowing ---
WINDOW_SIZE    = 72

# --- Split ---
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# --- Training ---
BATCH_SIZE     = 128
LSTM_UNITS     = 64
NUM_HEADS      = 4
KEY_DIM        = LSTM_UNITS // NUM_HEADS
LEARNING_RATE  = 1e-3
EPOCHS_MAIN    = 25                
EPOCHS_DEMO    = 6                 
PATIENCE_ES    = 4                 
PATIENCE_LR    = 2                 

# --- Quick mode for pipeline testing ---
QUICK_RUN      = False
if QUICK_RUN:
    EPOCHS_MAIN, EPOCHS_DEMO = 2, 1

# --- Performance target ---
MAE_GATE_SCALED = 0.015            

SUBMISSION_DIR = f"DLTM_{STUDENT_NAME}"
print(f"Horizon             : {HORIZON} langkah")
print(f"Target              : {TARGET}")
print(f"Split               : {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%} (kronologis)")
print(f"Gate MAE (scaled)   : < {MAE_GATE_SCALED}")
print(f"Nama Folder   : {SUBMISSION_DIR}")

# %% [markdown]
# ---
# # 2. Dataset Loading

# %%
# "Multivariate Crypto Data Hourly since 2017 to 2023"
csv_url = 'https://drive.google.com/uc?export=download&id=1hpsqSpfjdqIZWqwd259klQSeaNSe5Trr'

try:
    df_raw = pd.read_csv(csv_url)
    print(f"Dataset berhasil dimuat dari URL resmi. Shape: {df_raw.shape}")
except Exception as e:
    
    # Fallback: file lokal
    print(f"Gagal memuat dari URL ({type(e).__name__}). Mencoba file lokal 'Bitcoin3.csv'...")
    df_raw = pd.read_csv("Bitcoin3.csv")
    print(f"Dataset berhasil dimuat dari file lokal. Shape: {df_raw.shape}")

df_raw.head()

# %%
print("=" * 24)
print("STRUKTUR DATASET MENTAH")
print("=" * 24)
df_raw.info()
print("\nJumlah baris   :", f"{df_raw.shape[0]:,}")
print("Jumlah kolom   :", df_raw.shape[1])
print("Nama kolom     :", list(df_raw.columns))

# %% [markdown]
# ---
# # 3. Exploratory Data Analysis (EDA)
# 
# ## 3.1 Temporal Audit

# %%
df = df_raw.copy()

# --- (1) Parsing timestamp ---
df["Date"] = pd.to_datetime(df["Date"], format="ISO8601")

# --- (2) Grid alignment audit ---
off_grid = df[(df["Date"].dt.minute != 0) | (df["Date"].dt.second != 0) |
              (df["Date"].dt.microsecond != 0)]
print(f"Timestamp TIDAK tepat di awal jam : {len(off_grid):,} baris")
if len(off_grid):
    print("Contoh timestamp menyimpang:")
    print(off_grid["Date"].head(3).to_string(index=False))

# --- (3) Order & duplication ---
print(f"\nIndeks waktu monoton naik        : {df['Date'].is_monotonic_increasing}")
print(f"Timestamp duplikat               : {df['Date'].duplicated().sum():,}")
print(f"Baris duplikat penuh             : {df.duplicated().sum():,}")
print(f"Missing value per kolom          :\n{df.isna().sum().to_string()}")
print(f"\nRentang waktu : {df['Date'].min()}  ->  {df['Date'].max()}")
print(f"Rentang hari  : {(df['Date'].max() - df['Date'].min()).days:,} hari")

# %%
# --- Grid normalization ---
df["Date"] = df["Date"].dt.floor("h")
n_before = len(df)
df = df.drop_duplicates(subset="Date", keep="first").sort_values("Date").set_index("Date")
print(f"Baris setelah floor + dedup : {len(df):,} (dari {n_before:,})")

# --- Audit of the completeness of the hour grid ---
full_index = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
missing_hours = len(full_index) - len(df)
print(f"Grid jam ideal             : {len(full_index):,} baris")
print(f"Jam yang hilang (gap)      : {missing_hours:,}  ({missing_hours/len(full_index):.3%} dari rentang)")

deltas = pd.Series(df.index).diff().dropna()
print("\nDistribusi selisih antar-observasi (top 6):")
print(deltas.value_counts().head(6).to_string())

# %% [markdown]
# ### 3.2 Gap Handling

# %%
df = df.reindex(full_index)
n_interp = int(df.isna().any(axis=1).sum())
df = df.interpolate(method="time").ffill().bfill()
df.index.name = "Date"

print(f"Baris diinterpolasi   : {n_interp:,}")
print(f"Shape setelah reindex : {df.shape}")
print(f"Sisa NaN              : {int(df.isna().sum().sum())}")
print(f"Frekuensi indeks      : {pd.infer_freq(df.index)}")
df.head(3)

# %% [markdown]
# ### 3.3 Descriptive Statistics

# %%
desc = df.describe().T[["min", "mean", "max", "std"]]
desc["range"] = desc["max"] - desc["min"]
desc["orde_magnitudo"] = np.log10(desc["range"].abs().replace(0, np.nan)).round(2)
display(desc.style.format("{:,.4f}").set_caption(
    "Statistik deskriptif — perhatikan rentang 'Volume USDT' vs 'RSI'"))

print("\nRasio rentang terbesar terhadap terkecil: "
      f"{desc['range'].max() / desc['range'].min():,.0f}x")
print("\nInterpretasi: tanpa normalisasi per-kolom, kontribusi gradien akan didominasi")
print("oleh 'Volume USDT'. Fitur bounded seperti RSI praktis tidak akan berpengaruh.")

# %%
fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
for ax, col in zip(axes.ravel(), df.columns):
    ax.plot(df.index, df[col], lw=0.4, color="#2b6cb0")
    ax.set_title(col, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
fig.suptitle("Deret Waktu Seluruh Variabel — Bitcoin Hourly (2017–2023)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # 4. Feature Engineering — Rolling Statistic

# %%
ROLL_WINDOW = 24

df["Close_roll_mean_24"]  = df[TARGET].rolling(window=ROLL_WINDOW).mean()
df["Close_roll_std_24"]   = df[TARGET].rolling(window=ROLL_WINDOW).std()
df["Volume_roll_mean_24"] = df["Volume USDT"].rolling(window=ROLL_WINDOW).mean()

n_before = len(df)
df = df.dropna()   
print(f"Baris dibuang akibat warm-up rolling : {n_before - len(df)}")
print(f"Shape akhir                          : {df.shape}")


FEATURES = [TARGET, "Volume USDT", "RSI", "MACD_Hist", "ATR", "KAMAO",
            "Close_roll_mean_24", "Close_roll_std_24", "Volume_roll_mean_24"]
N_FEATURES     = len(FEATURES)
TARGET_COL_IDX = FEATURES.index(TARGET)

print(f"\nJumlah fitur input model : {N_FEATURES}  (kriteria minimum: 3 -> TERPENUHI)")
print(f"Daftar fitur             : {FEATURES}")
print(f"Indeks kolom target      : {TARGET_COL_IDX}")
df[FEATURES].tail(3)

# %% [markdown]
# ## 4.1 Feature-to-Feature Correlation Heatmap

# %%
corr = df[FEATURES].corr()

fig, ax = plt.subplots(figsize=(10.5, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=.6, cbar_kws={"shrink": .8},
            annot_kws={"size": 8}, ax=ax)
ax.set_title("Heatmap Korelasi Pearson Antar-Fitur Terpilih\n(Bitcoin Hourly, 2017–2023)",
             fontsize=12, fontweight="bold", pad=14)
plt.tight_layout()
plt.show()

# %%
target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#2f855a" if v > 0 else "#c53030" for v in target_corr.values]
ax.barh(target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1])
ax.axvline(0, color="black", lw=.8)
ax.set_xlabel("Korelasi Pearson terhadap Close")
ax.set_title("Korelasi Setiap Fitur terhadap Target (Close)", fontweight="bold")
plt.tight_layout(); plt.show()

print(target_corr.round(4).to_string())

# %% [markdown]
# ### 4.2 Heatmap Interpretation
# 
# **Main Findings:**
# 
# 1. **Level-based features show strong correlation with the target.** `Close_roll_mean_24` (=0,99) and
#    `ATR` (=0,84) move in tandem with the price level. The correlation of `Close_roll_mean_24` that
#    approaches perfect is a mathematical consequence of its construction (a rolling average
#    of the target itself) — this is **not** data leakage, as its window is *trailing*.
# 
# 2. **The oscillator has a linear correlation of ≈ 0 with price levels.** `RSI` (=0.04), `KAMAO` (=0.02), 
# and `MACD_Hist` (=-0.02) seem 'uninformative' if judged just from the heatmap..
# 
# 3. **Multicollinearity between oscillators.** `RSI`–`KAMAO` (=0.75) and `KAMAO`–`MACD_Hist` (=0.50) 
# show partial redundancy. In linear regression, this would inflate the *standard error*. 
# The consequence is accepted consciously.

# %% [markdown]
# ---
# # 5. Decomposition Analysis
# 
# ## 5.1 Augmented Dickey-Fuller (ADF) Testing

# %%
def adf_report(series, name):
    r = adfuller(series.dropna().values, autolag="AIC")
    verdict = "STASIONER (tolak H0)" if r[1] < 0.05 else "NON-STASIONER (gagal tolak H0)"
    print(f"{name:<28s} ADF stat = {r[0]:10.4f} | p-value = {r[1]:.4g} | "
          f"lag terpilih = {r[2]:2d} | -> {verdict}")
    return r[1]

print("=" * 68)
print("UJI AUGMENTED DICKEY-FULLER (H0: terdapat unit root / non-stasioner)")
print("=" * 68)
p_level = adf_report(df[TARGET], "Close (level)")
p_diff  = adf_report(df[TARGET].diff(), "ΔClose (first difference)")
for c in ["RSI", "MACD_Hist", "KAMAO", "ATR"]:
    adf_report(df[c], c)

# %% [markdown]
# ### 5.2 Implications of the ADF Results
# 
# The results above show the classic pattern of asset price series:
# 
# - **`Close` (level): NON-STASIONER** (p = 0,50 → failed to reject H₀).
# - **`ΔClose` (first difference): STASIONER** (p = 0,000 → reject H₀).

# %% [markdown]
# ## 5.3 Decomposition of the Target Series

# %%
# --- Decomposition of the full range ---
decomp_full = seasonal_decompose(df[TARGET], model="additive", period=24,
                                 extrapolate_trend="freq")

fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
for ax, (comp, ttl, col) in zip(axes, [
        (decomp_full.observed, "Observed — Close", "#1a365d"),
        (decomp_full.trend,    "Trend",            "#2f855a"),
        (decomp_full.seasonal, "Seasonal (period = 24 jam)", "#b7791f"),
        (decomp_full.resid,    "Residual",         "#9b2c2c")]):
    ax.plot(comp.index, comp.values, lw=0.4, color=col)
    ax.set_ylabel(ttl, fontsize=11, fontweight="bold")
fig.suptitle("Dekomposisi Aditif — Close Bitcoin (Rentang Penuh, period = 24)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

var_total = np.nanvar(decomp_full.observed)
print(f"Proporsi varians komponen musiman terhadap total : "
      f"{np.nanvar(decomp_full.seasonal)/var_total:.6%}")
print(f"Proporsi varians komponen residual terhadap total: "
      f"{np.nanvar(decomp_full.resid)/var_total:.6%}")
print(f"Proporsi varians komponen tren terhadap total    : "
      f"{np.nanvar(decomp_full.trend)/var_total:.6%}")

# %%
# --- Decomposition of the subsample (30 days) ---
sub = df[TARGET].iloc[-24 * 30:]
decomp_sub = seasonal_decompose(sub, model="additive", period=24, extrapolate_trend="freq")

fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
for ax, (comp, ttl, col) in zip(axes, [
        (decomp_sub.observed, "Observed", "#1a365d"),
        (decomp_sub.trend,    "Trend",    "#2f855a"),
        (decomp_sub.seasonal, "Seasonal (period = 24 jam)", "#b7791f"),
        (decomp_sub.resid,    "Residual", "#9b2c2c")]):
    ax.plot(comp.index, comp.values, lw=1.0, color=col)
    ax.set_ylabel(ttl, fontsize=9, fontweight="bold")
fig.suptitle("Dekomposisi Aditif — Close Bitcoin (30 Hari Terakhir, period = 24)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# --- Profil musiman rata-rata per jam ---
seasonal_profile = decomp_full.seasonal.groupby(decomp_full.seasonal.index.hour).mean()
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.bar(seasonal_profile.index, seasonal_profile.values, color="#b7791f")
ax.axhline(0, color="black", lw=.8)
ax.set_xlabel("Jam (UTC)"); ax.set_ylabel("Efek musiman rata-rata (USD)")
ax.set_title("Profil Musiman Intraday (period = 24)", fontweight="bold")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 5.4 Interpretation of Decomposition
# 
# 1. **Trend dominates the total variance** (>99%). In other words, the visual confirmation of the ADF results: the series is controlled by a non-stationary stochastic component.
# 2. **The intraday seasonal component exists, but its amplitude is very small** relative to
#    the price level. The daily structure remains acceptable for retention as an encoder context, but **is not sufficient** as a standalone prediction basis.
# 3. **The residual shows clear heteroskedasticity** — the variance spreads significantly in 2021 (high prices). This is the classic *volatility clustering*, and precisely why `ATR` and `Close_roll_std_24` are retained as features: both provide explicit information about the current volatility state.

# %% [markdown]
# ---
# # 6. ACF & PACF Testing — Determining WINDOW SIZE

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 7))

plot_acf(df[TARGET].values,  lags=96, ax=axes[0, 0], alpha=0.05)
axes[0, 0].set_title("ACF — Close (level, NON-stasioner)", fontweight="bold")

plot_pacf(df[TARGET].values, lags=48, ax=axes[0, 1], alpha=0.05, method="ywm")
axes[0, 1].set_title("PACF — Close (level, NON-stasioner)", fontweight="bold")

d_close = df[TARGET].diff().dropna().values
plot_acf(d_close,  lags=96, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title("ACF — ΔClose (first difference, STASIONER)", fontweight="bold")

plot_pacf(d_close, lags=48, ax=axes[1, 1], alpha=0.05, method="ywm")
axes[1, 1].set_title("PACF — ΔClose (first difference, STASIONER)", fontweight="bold")

for ax in axes.ravel():
    ax.set_xlabel("Lag (jam)", fontsize=9)
fig.suptitle("Analisis Lag: ACF & PACF pada Level dan First Difference",
             fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

# %%
from statsmodels.tsa.stattools import acf as acf_vals, pacf as acf_pvals

n_obs   = len(df)
ci_band = 1.96 / np.sqrt(n_obs)          

acf_level  = acf_vals(df[TARGET].values, nlags=96, fft=True)
pacf_level = acf_pvals(df[TARGET].values, nlags=48, method="ywm")
acf_diff   = acf_vals(d_close, nlags=96, fft=True)
pacf_diff  = acf_pvals(d_close, nlags=48, method="ywm")

print("=" * 60)
print(f"ANALISIS LAG  |  n = {n_obs:,}  |  signifikansi 95% = ±{ci_band:.5f}")
print("=" * 60)

probe = [1, 6, 12, 24, 48, 72, 96]
print("\n[A] ACF pada LEVEL (Close):")
for L in probe:
    print(f"    lag {L:3d} : {acf_level[L]: .4f}")
print("    -> Peluruhan sangat lambat (masih >0,99 pada lag 96). Ciri khas proses I(1);")
print("       ACF level TIDAK dapat dipakai untuk memilih window (spurious).")

print("\n[B] PACF pada LEVEL (Close):")
for L in [1, 2, 3, 4]:
    print(f"    lag {L:3d} : {pacf_level[L]: .4f}")
print("    -> Spike tunggal dominan di lag 1 lalu menurun ke ~0.")
print("       Level harga hampir sepenuhnya dijelaskan oleh harga satu jam sebelumnya.")

sig_pacf_diff = [L for L in range(1, 49) if abs(pacf_diff[L]) > ci_band]
sig_acf_diff  = [L for L in range(1, 97) if abs(acf_diff[L])  > ci_band]
print("\n[C] Struktur lag SIGNIFIKAN pada ΔClose (deret stasioner):")
print(f"    PACF signifikan pada lag : {sig_pacf_diff}")
print(f"    ACF  signifikan pada lag : {sig_acf_diff[:20]}{' ...' if len(sig_acf_diff)>20 else ''}")
print(f"    Lag signifikan TERJAUH pada PACF: {max(sig_pacf_diff) if sig_pacf_diff else 0}")
print(f"    Lag 24 (siklus harian) signifikan? "
      f"{'YA' if abs(pacf_diff[24]) > ci_band else 'tidak'}  (PACF[24] = {pacf_diff[24]:+.5f})")

# %% [markdown]
# ### 6.1 Setting the `WINDOW_SIZE`

# %%
# --- Final window size setting ---
max_sig_lag  = max(sig_pacf_diff) if sig_pacf_diff else 24
WINDOW_SIZE  = 72   

print("=" * 45)
print("PENETAPAN WINDOW SIZE (berbasis Uji ACF/PACF)")
print("=" * 45)
print(f"Lag PACF signifikan terjauh (ΔClose) : {max_sig_lag}")
print(f"Periode musiman (dekomposisi)        : 24 jam")
print(f"Horizon prediksi (wajib)             : {HORIZON} langkah")
print(f"WINDOW_SIZE FINAL                    : {WINDOW_SIZE} jam  (= 3 x 24)")
print(f"  - mencakup lag signifikan ≤ {max_sig_lag}?      {'YA' if WINDOW_SIZE >= max_sig_lag else 'TIDAK'}"
      f"  (margin {WINDOW_SIZE - max_sig_lag} lag)")
print(f"  - kelipatan periode musiman 24?    {'YA' if WINDOW_SIZE % 24 == 0 else 'TIDAK'}"
      f"  ({WINDOW_SIZE // 24} siklus harian)")
print(f"  - rasio window : horizon           = {WINDOW_SIZE // HORIZON} : 1")

# %% [markdown]
# ---
# # 7. Data Splitting & Normalization
# 
# ## 7.1 Per-Column Normalization

# %%
# ========== CHRONOLOGICAL SPLIT ===========
n_total = len(df)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)

df_train = df.iloc[:n_train].copy()
df_val   = df.iloc[n_train:n_train + n_val].copy()
df_test  = df.iloc[n_train + n_val:].copy()

split_tbl = pd.DataFrame([
    {"Partisi": "TRAIN", "Baris": len(df_train), "Proporsi": len(df_train)/n_total,
     "Mulai": df_train.index.min(), "Selesai": df_train.index.max(),
     "Close min": df_train[TARGET].min(), "Close max": df_train[TARGET].max()},
    {"Partisi": "VALIDATION", "Baris": len(df_val), "Proporsi": len(df_val)/n_total,
     "Mulai": df_val.index.min(), "Selesai": df_val.index.max(),
     "Close min": df_val[TARGET].min(), "Close max": df_val[TARGET].max()},
    {"Partisi": "TEST", "Baris": len(df_test), "Proporsi": len(df_test)/n_total,
     "Mulai": df_test.index.min(), "Selesai": df_test.index.max(),
     "Close min": df_test[TARGET].min(), "Close max": df_test[TARGET].max()},
])
display(split_tbl.style.format({"Baris": "{:,}", "Proporsi": "{:.1%}",
                                "Close min": "{:,.0f}", "Close max": "{:,.0f}"})
        .set_caption("Pembagian data kronologis — tiga partisi (train / validation / test)"))

assert df_train.index.max() < df_val.index.min() < df_test.index.min(), "Urutan partisi rusak!"
print("VERIFIKASI: urutan temporal train < validation < test  ->  OK (tidak ada tumpang tindih)")

# %%
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df_train.index, df_train[TARGET], lw=.5, color="#2b6cb0", label=f"Train ({len(df_train):,})")
ax.plot(df_val.index,   df_val[TARGET],   lw=.5, color="#b7791f", label=f"Validation ({len(df_val):,})")
ax.plot(df_test.index,  df_test[TARGET],  lw=.5, color="#c53030", label=f"Test ({len(df_test):,})")
for b in [df_val.index.min(), df_test.index.min()]:
    ax.axvline(b, color="black", ls="--", lw=1)
ax.set_ylabel("Close (USD)"); ax.set_xlabel("Waktu")
ax.set_title("Pembagian Kronologis Data — Batas Partisi",
             fontweight="bold")
ax.legend(); plt.tight_layout(); plt.show()

# %%
# ============ FIT SCALER ============
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler  = MinMaxScaler(feature_range=(0, 1))

feature_scaler.fit(df_train[FEATURES])      # <-- fit HANYA pada df_train
target_scaler.fit(df_train[[TARGET]])       # <-- scaler target terpisah utk inverse transform

# ============ TRANSFORM ============
train_scaled = feature_scaler.transform(df_train[FEATURES]).astype("float32")
val_scaled   = feature_scaler.transform(df_val[FEATURES]).astype("float32")
test_scaled  = feature_scaler.transform(df_test[FEATURES]).astype("float32")

print("=" * 73)
print("BUKTI ANTI-LEAKAGE: parameter scaler diturunkan HANYA dari partisi TRAIN")
print("=" * 73)
proof = pd.DataFrame({
    "scaler data_min_ (dari TRAIN)": feature_scaler.data_min_,
    "scaler data_max_ (dari TRAIN)": feature_scaler.data_max_,
    "min AKTUAL train": df_train[FEATURES].min().values,
    "max AKTUAL train": df_train[FEATURES].max().values,
    "min aktual test (TIDAK dipakai)": df_test[FEATURES].min().values,
    "max aktual test (TIDAK dipakai)": df_test[FEATURES].max().values,
}, index=FEATURES)
display(proof.style.format("{:,.4f}").set_caption(
    "Parameter scaler identik dengan statistik TRAIN, dan berbeda dari statistik TEST"))

assert np.allclose(feature_scaler.data_min_, df_train[FEATURES].min().values), "Leakage!"
assert np.allclose(feature_scaler.data_max_, df_train[FEATURES].max().values), "Leakage!"
print("ASSERTION LULUS: data_min_/data_max_ scaler == statistik TRAIN, bukan statistik gabungan.")
print("                 -> Tidak ada informasi validation/test yang masuk ke proses training.\n")

print(f"Rentang target dari TRAIN : [{target_scaler.data_min_[0]:,.2f} , "
      f"{target_scaler.data_max_[0]:,.2f}]  (span = {target_scaler.data_max_[0]-target_scaler.data_min_[0]:,.2f} USD)")
print(f"Close TEST setelah di-scale : [{test_scaled[:, TARGET_COL_IDX].min():.4f} , "
      f"{test_scaled[:, TARGET_COL_IDX].max():.4f}]")
print("Catatan: nilai test berada di dalam [0,1] karena rentang harga periode uji tercakup")
print("         dalam rentang periode latih. Ini diverifikasi, bukan diasumsikan.")

# Verifikasi bahwa kolom target pada feature_scaler identik dengan target_scaler
assert np.isclose(feature_scaler.data_min_[TARGET_COL_IDX], target_scaler.data_min_[0])
assert np.isclose(feature_scaler.data_max_[TARGET_COL_IDX], target_scaler.data_max_[0])
print("\nVERIFIKASI: kolom Close pada feature_scaler == target_scaler (level anchor konsisten).")

# %% [markdown]
# ### 7.3 Batas Interpretasi MAE Gate

# %%
span_usd = target_scaler.data_max_[0] - target_scaler.data_min_[0]
print(f"Gate {MAE_GATE_SCALED} (scaled)  ==  {MAE_GATE_SCALED * span_usd:,.0f} USD galat absolut rata-rata\n")

# ---- Benchmark 1: PERSISTENCE (random-walk) — lower limit ----
tgt_test = test_scaled[:, TARGET_COL_IDX]
idx      = np.arange(0, len(tgt_test) - WINDOW_SIZE - HORIZON)
anchor   = tgt_test[idx + WINDOW_SIZE - 1][:, None]
y_true_p = np.stack([tgt_test[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON] for i in idx])
mae_persistence = float(np.mean(np.abs(y_true_p - anchor)))

# ---- Benchmark 2: MEAN-REVERSION DEGENERATIF ----
mae_degenerate = float(np.mean(np.abs(y_true_p - train_scaled[:, TARGET_COL_IDX].mean())))

bench = pd.DataFrame([
    {"Benchmark": "Persistence / random-walk (anchor = harga terakhir)",
     "MAE (scaled)": mae_persistence, "MAE (USD)": mae_persistence * span_usd,
     "vs Gate": "LULUS" if mae_persistence < MAE_GATE_SCALED else "GAGAL"},
    {"Benchmark": f"GATE KRITERIA ADVANCED", "MAE (scaled)": MAE_GATE_SCALED,
     "MAE (USD)": MAE_GATE_SCALED * span_usd, "vs Gate": "—"},
    {"Benchmark": "Degeneratif (prediksi = rata-rata train)",
     "MAE (scaled)": mae_degenerate, "MAE (USD)": mae_degenerate * span_usd,
     "vs Gate": "LULUS" if mae_degenerate < MAE_GATE_SCALED else "GAGAL"},
])
display(bench.style.format({"MAE (scaled)": "{:.5f}", "MAE (USD)": "{:,.0f}"})
        .set_caption("Benchmark referensi untuk gate MAE < 0,015 (ruang ter-skala)"))

print(f"Rasio gate terhadap persistence: {MAE_GATE_SCALED / mae_persistence:.1f}x")
print("\nInterpretasi:")
print(f"  - Model yang GAGAL menangkap level (regresi ke rata-rata train) -> MAE ≈ {mae_degenerate:.3f},")
print(f"    yaitu ~{mae_degenerate/MAE_GATE_SCALED:.0f}x DI ATAS gate. Inilah mode kegagalan utama yang")
print("    diantisipasi oleh desain level-anchored residual decoder (Bagian 5.2).")
print(f"  - Persistence sudah mencapai MAE ≈ {mae_persistence:.5f}. Karena itu gate 0,015 dapat")
print("    dicapai; syaratnya model harus ter-anchor pada level, bukan pada rata-rata global.")

# %% [markdown]
# ---
# # 8. Windowing and Pipeline
# 
# ## 8.1 Tensor

# %%
def create_windows(scaled_array, window_size, horizon, target_idx):

    n_samples = len(scaled_array) - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError("Partisi terlalu pendek untuk window_size + horizon yang diminta.")

    tgt = scaled_array[:, target_idx]

    enc = np.stack([scaled_array[i : i + window_size]
                    for i in range(n_samples)]).astype("float32")
    y   = np.stack([tgt[i + window_size : i + window_size + horizon]
                    for i in range(n_samples)]).astype("float32")[..., None]

    # decoder_input 
    last_obs = tgt[window_size - 1 : window_size - 1 + n_samples].astype("float32")[:, None, None]
    dec = np.concatenate([last_obs, y[:, :-1, :]], axis=1)

    return enc, dec, y


enc_train, dec_train, y_train = create_windows(train_scaled, WINDOW_SIZE, HORIZON, TARGET_COL_IDX)
enc_val,   dec_val,   y_val   = create_windows(val_scaled,   WINDOW_SIZE, HORIZON, TARGET_COL_IDX)
enc_test,  dec_test,  y_test  = create_windows(test_scaled,  WINDOW_SIZE, HORIZON, TARGET_COL_IDX)

shapes = pd.DataFrame([
    {"Partisi": "TRAIN", "encoder_input": str(enc_train.shape),
     "decoder_input": str(dec_train.shape), "y": str(y_train.shape)},
    {"Partisi": "VALIDATION", "encoder_input": str(enc_val.shape),
     "decoder_input": str(dec_val.shape), "y": str(y_val.shape)},
    {"Partisi": "TEST", "encoder_input": str(enc_test.shape),
     "decoder_input": str(dec_test.shape), "y": str(y_test.shape)},
])
display(shapes.set_index("Partisi").style.set_caption(
    f"Tensor setelah windowing (WINDOW_SIZE={WINDOW_SIZE}, HORIZON={HORIZON})"))

# --- Verifikasi konstruksi teacher forcing pada satu sampel ---
s = 0
print("VERIFIKASI TEACHER FORCING (partisi train):")
print(f"  encoder_input[-1, Close]        = {enc_train[s, -1, TARGET_COL_IDX]:.6f}   <- y_t (observasi terakhir)")
print(f"  decoder_input[0]               = {dec_train[s, 0, 0]:.6f}   <- harus == y_t")
print(f"  decoder_input[1:4]             = {dec_train[s, 1:4, 0].round(6)}")
print(f"  y[0:3]                         = {y_train[s, 0:3, 0].round(6)}   <- decoder_input[1:4] adalah y[0:3]")
assert np.allclose(dec_train[:, 0, 0], enc_train[:, -1, TARGET_COL_IDX]), "Anchor decoder salah"
assert np.allclose(dec_train[:, 1:, 0], y_train[:, :-1, 0]),               "Pergeseran teacher forcing salah"
print("\nASSERTION LULUS: decoder_input == [y_t, y_{t+1}, ..., y_{t+23}] (geser satu langkah).")

# %%
AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(enc, dec, y, batch_size=BATCH_SIZE, shuffle=False, seq2seq=True):

    inputs = (enc, dec) if seq2seq else enc
    ds = tf.data.Dataset.from_tensor_slices((inputs, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(y), 10_000), seed=SEED,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

# --- Pipeline Seq2Seq (two input) ---
train_ds_s2s = make_dataset(enc_train, dec_train, y_train, shuffle=True,  seq2seq=True)
val_ds_s2s   = make_dataset(enc_val,   dec_val,   y_val,   shuffle=False, seq2seq=True)
test_ds_s2s  = make_dataset(enc_test,  dec_test,  y_test,  shuffle=False, seq2seq=True)

# --- Pipeline baseline (one input) ---
train_ds_base = make_dataset(enc_train, dec_train, y_train, shuffle=True,  seq2seq=False)
val_ds_base   = make_dataset(enc_val,   dec_val,   y_val,   shuffle=False, seq2seq=False)
test_ds_base  = make_dataset(enc_test,  dec_test,  y_test,  shuffle=False, seq2seq=False)

print("=" * 33)
print("PIPELINE — TIGA PARTISI TERPISAH")
print("=" * 33)
for nm, ds, n in [("TRAIN", train_ds_s2s, len(y_train)),
                  ("VALIDATION", val_ds_s2s, len(y_val)),
                  ("TEST", test_ds_s2s, len(y_test))]:
    print(f"{nm:<11s} | {n:>7,} sampel | {len(ds):>4} batch | element_spec OK")

(xb_enc, xb_dec), yb = next(iter(train_ds_s2s))
print(f"\nBatch Seq2Seq  : encoder {tuple(xb_enc.shape)} | "
      f"decoder {tuple(xb_dec.shape)} | y {tuple(yb.shape)}")
xb_b, yb_b = next(iter(train_ds_base))
print(f"Batch baseline : encoder {tuple(xb_b.shape)} | y {tuple(yb_b.shape)}")
print("\nOptimasi: .batch() -> .prefetch(AUTOTUNE); shuffle hanya pada TRAIN.")

# %% [markdown]
# ---
# # 9. Custom Layers — Implementation from Scratch
# 
# ## 9.1 Reimplementation of Dense Layer

# %%
@keras.saving.register_keras_serializable(package="DLTM")
class CustomDense(layers.Layer):

    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units          = int(units)
        self.activation     = keras.activations.get(activation)
        self.use_bias       = bool(use_bias)
        self._activation_id = activation

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        self.w = self.add_weight(
            name="kernel", shape=(d_in, self.units),
            initializer=keras.initializers.GlorotUniform(seed=SEED), trainable=True)
        if self.use_bias:
            self.b = self.add_weight(
                name="bias", shape=(self.units,),
                initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        y = tf.matmul(inputs, self.w)          # kontraksi pada sumbu terakhir -> rank-agnostic
        if self.use_bias:
            y = y + self.b
        return self.activation(y)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units,
                    "activation": keras.activations.serialize(self.activation),
                    "use_bias": self.use_bias})
        return cfg

# %% [markdown]
# ## 9.2 Reimplementation of Multi-Head Attention

# %%
@keras.saving.register_keras_serializable(package="DLTM")
class CustomMultiHeadAttention(layers.Layer):

    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads    = int(num_heads)
        self.key_dim      = int(key_dim)
        self.dropout_rate = float(dropout)

    def build(self, input_shape):
        q_shape, v_shape = input_shape[0], input_shape[1]
        self.d_q   = int(q_shape[-1])
        d_v        = int(v_shape[-1])
        inner      = self.num_heads * self.key_dim
        init       = keras.initializers.GlorotUniform(seed=SEED)

        self.wq = self.add_weight(name="query_kernel",  shape=(self.d_q, inner), initializer=init, trainable=True)
        self.wk = self.add_weight(name="key_kernel",    shape=(d_v,      inner), initializer=init, trainable=True)
        self.wv = self.add_weight(name="value_kernel",  shape=(d_v,      inner), initializer=init, trainable=True)
        self.wo = self.add_weight(name="output_kernel", shape=(inner, self.d_q), initializer=init, trainable=True)
        self.bo = self.add_weight(name="output_bias",   shape=(self.d_q,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def _split_heads(self, x, batch, length):

        x = tf.reshape(x, (batch, length, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        query, value = inputs[0], inputs[1]
        key   = value                                   
        batch = tf.shape(query)[0]
        t_q   = tf.shape(query)[1]
        t_v   = tf.shape(value)[1]

        # Linear Projection
        q = self._split_heads(tf.matmul(query, self.wq), batch, t_q)   # (B, H, Tq, dk)
        k = self._split_heads(tf.matmul(key,   self.wk), batch, t_v)   # (B, H, Tv, dk)
        v = self._split_heads(tf.matmul(value, self.wv), batch, t_v)   # (B, H, Tv, dk)

        # Scaled dot-product attention
        scale  = tf.math.sqrt(tf.cast(self.key_dim, q.dtype))
        scores = tf.matmul(q, k, transpose_b=True) / scale             # (B, H, Tq, Tv)
        weights = tf.nn.softmax(scores, axis=-1)
        if training and self.dropout_rate > 0.0:
            weights = tf.nn.dropout(weights, rate=self.dropout_rate)
        context = tf.matmul(weights, v)                                # (B, H, Tq, dk)

        # Gabungkan head + proyeksi keluaran
        context = tf.transpose(context, perm=[0, 2, 1, 3])             # (B, Tq, H, dk)
        context = tf.reshape(context, (batch, t_q, self.num_heads * self.key_dim))
        return tf.matmul(context, self.wo) + self.bo                   # (B, Tq, d_q)

    def compute_output_shape(self, input_shape):
        q_shape = input_shape[0]
        return (q_shape[0], q_shape[1], q_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads, "key_dim": self.key_dim,
                    "dropout": self.dropout_rate})
        return cfg

# %% [markdown]
# ## 9.3 `CustomDropout`, `CustomLayerNormalization`, `CustomLeakyReLU`

# %%
@keras.saving.register_keras_serializable(package="DLTM")
class CustomDropout(layers.Layer):

    def __init__(self, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def call(self, inputs, training=False):
        if (not training) or self.rate <= 0.0:
            return inputs                                   # identitas saat inferensi
        keep_prob = 1.0 - self.rate
        mask = tf.cast(
            tf.random.uniform(tf.shape(inputs), minval=0.0, maxval=1.0) < keep_prob,
            inputs.dtype)
        return (inputs * mask) / keep_prob                  # inverted: skala saat LATIH

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config(); cfg.update({"rate": self.rate}); return cfg


@keras.saving.register_keras_serializable(package="DLTM")
class CustomLayerNormalization(layers.Layer):

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(d,), initializer="ones",  trainable=True)
        self.beta  = self.add_weight(name="beta",  shape=(d,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[-1], keepdims=True)   
        normalized = (inputs - mean) * tf.math.rsqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config(); cfg.update({"epsilon": self.epsilon}); return cfg


@keras.saving.register_keras_serializable(package="DLTM")
class CustomLeakyReLU(layers.Layer):

    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def call(self, inputs):
        return tf.maximum(self.alpha * inputs, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config(); cfg.update({"alpha": self.alpha}); return cfg

# %% [markdown]
# ## 9.4 Operationalization of ADF Test Findings

# %%
@keras.saving.register_keras_serializable(package="DLTM")
class LevelAnchorAdd(layers.Layer):

    def __init__(self, target_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.target_idx = int(target_idx)

    def call(self, inputs):
        delta, encoder_input = inputs[0], inputs[1]
        # harga penutupan terakhir yang teramati -> (B, 1, 1)
        anchor = encoder_input[:, -1:, self.target_idx : self.target_idx + 1]
        if len(delta.shape) == 2:                       
            anchor = tf.squeeze(anchor, axis=1)        
        return delta + anchor                           

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        cfg = super().get_config(); cfg.update({"target_idx": self.target_idx}); return cfg

# %% [markdown]
# ## 9.5 Custom Layer Numerical Validation

# %%
print("=" * 45)
print("VALIDASI NUMERIK CUSTOM LAYER TERHADAP KERAS")
print("=" * 45)

rng = np.random.default_rng(SEED)

# ---------- (1) CustomDense vs layers.Dense ----------
x = tf.constant(rng.normal(size=(4, 24, 16)).astype("float32"))
cd, kd = CustomDense(8, activation="relu"), layers.Dense(8, activation="relu")
_, _ = cd(x), kd(x)
kd.set_weights([cd.w.numpy(), cd.b.numpy()])              # sinkronkan bobot
err = float(tf.reduce_max(tf.abs(cd(x) - kd(x))))
print(f"[1] CustomDense              vs Dense                : max |Δ| = {err:.3e}  "
      f"{'LULUS' if err < 1e-5 else 'GAGAL'}   (output {tuple(cd(x).shape)}, rank-3 tanpa TimeDistributed)")

# ---------- (2) CustomLayerNormalization vs layers.LayerNormalization ----------
cln, kln = CustomLayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
_, _ = cln(x), kln(x)
err = float(tf.reduce_max(tf.abs(cln(x) - kln(x))))
print(f"[2] CustomLayerNormalization vs LayerNormalization   : max |Δ| = {err:.3e}  "
      f"{'LULUS' if err < 1e-4 else 'GAGAL'}")

# ---------- (3) CustomLeakyReLU vs layers.LeakyReLU ----------
clr, klr = CustomLeakyReLU(alpha=0.1), layers.LeakyReLU(negative_slope=0.1)
err = float(tf.reduce_max(tf.abs(clr(x) - klr(x))))
print(f"[3] CustomLeakyReLU          vs LeakyReLU            : max |Δ| = {err:.3e}  "
      f"{'LULUS' if err < 1e-6 else 'GAGAL'}")

# ---------- (4) Custom Dropout ----------
cdo = CustomDropout(rate=0.3)
identity_err = float(tf.reduce_max(tf.abs(cdo(x, training=False) - x)))
big  = tf.ones((20000,))
mean_train = float(tf.reduce_mean(cdo(big, training=True)))
print(f"[4] CustomDropout            : identitas saat inferensi max |Δ| = {identity_err:.3e}  "
      f"{'LULUS' if identity_err == 0.0 else 'GAGAL'}")
print(f"    CustomDropout            : E[x] saat latih = {mean_train:.4f} (target 1,0)      "
      f"{'LULUS' if abs(mean_train - 1.0) < 0.03 else 'GAGAL'}  <- inverted dropout benar")

# ---------- (5) CustomMultiHeadAttention: bentuk + properti softmax ----------
q = tf.constant(rng.normal(size=(4, 24, 32)).astype("float32"))
v = tf.constant(rng.normal(size=(4, 72, 32)).astype("float32"))
cmha, kmha = CustomMultiHeadAttention(num_heads=4, key_dim=8), layers.MultiHeadAttention(num_heads=4, key_dim=8)
out_c = cmha([q, v]); out_k = kmha(query=q, value=v)
shape_ok = tuple(out_c.shape) == tuple(out_k.shape) == (4, 24, 32)
print(f"[5] CustomMultiHeadAttention vs MultiHeadAttention   : bentuk {tuple(out_c.shape)} == "
      f"{tuple(out_k.shape)}  {'LULUS' if shape_ok else 'GAGAL'}")
print(f"    (bobot berbeda -> nilai berbeda; yang divalidasi adalah kontrak bentuk & jalur komputasi)")
n_params_c = sum(int(np.prod(w.shape)) for w in cmha.trainable_weights)
n_params_k = sum(int(np.prod(w.shape)) for w in kmha.trainable_weights)
print(f"    Jumlah parameter: custom = {n_params_c:,} | keras = {n_params_k:,}")

# ---------- (6) LevelAnchorAdd: Close terakhir ----------
enc_probe   = tf.constant(rng.normal(size=(4, WINDOW_SIZE, N_FEATURES)).astype("float32"))
delta_probe = tf.zeros((4, HORIZON, 1))
anchored    = LevelAnchorAdd(target_idx=TARGET_COL_IDX)([delta_probe, enc_probe])
expected    = enc_probe[:, -1, TARGET_COL_IDX].numpy()
err = float(np.max(np.abs(anchored.numpy()[:, :, 0] - expected[:, None])))
print(f"[6] LevelAnchorAdd           : delta=0 -> output == Close terakhir, max |Δ| = {err:.3e}  "
      f"{'LULUS' if err < 1e-6 else 'GAGAL'}")
print("\nSeluruh custom layer tervalidasi secara numerik terhadap semantik referensinya.")

# %% [markdown]
# ---
# # 10. Custom Loss Functions
# ## 10.1 MAE
# 

# %%
# --------------------
# Custom Loss #1 — MAE
# --------------------
def custom_mae_loss(y_true, y_pred):

    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# ---------------------------------------
# Custom Loss #2 — Weighted MAE horizon 
# ---------------------------------------
class HorizonWeightedMAELoss:

    def __init__(self, horizon, w_start=1.0, w_step=0.2, normalize=True):
        self.horizon, self.w_start, self.w_step, self.normalize = horizon, w_start, w_step, normalize
        w = np.array([w_start + w_step * h for h in range(horizon)], dtype="float32")
        self.raw_weights = w.copy()
        if normalize:
            w = w / w.mean()                       
        self.weights_np = w
        self.weights    = tf.constant(w.reshape(1, horizon, 1), dtype=tf.float32)
        self.__name__   = "horizon_weighted_mae"

    def __call__(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        abs_err = tf.abs(y_true - y_pred)                        # (B, H, 1)
        return tf.reduce_mean(abs_err * tf.cast(self.weights, y_pred.dtype))


horizon_weighted_loss = HorizonWeightedMAELoss(HORIZON, w_start=1.0, w_step=0.2, normalize=True)

print("SKEMA BOBOT HORIZON (w_h = 1.0 + 0.2*(h-1)):")
wdf = pd.DataFrame({
    "Langkah (h)": np.arange(1, HORIZON + 1),
    "Bobot mentah w_h": horizon_weighted_loss.raw_weights,
    "Bobot ternormalisasi": horizon_weighted_loss.weights_np,
})
display(wdf.head(6).style.format({"Bobot mentah w_h": "{:.1f}", "Bobot ternormalisasi": "{:.4f}"})
        .hide(axis="index").set_caption("6 langkah pertama"))
print(f"Bobot mentah        : w_1={horizon_weighted_loss.raw_weights[0]:.1f}, "
      f"w_2={horizon_weighted_loss.raw_weights[1]:.1f}, w_3={horizon_weighted_loss.raw_weights[2]:.1f}, "
      f"..., w_{HORIZON}={horizon_weighted_loss.raw_weights[-1]:.1f}")
print(f"Jumlah bobot mentah : {horizon_weighted_loss.raw_weights.sum():.1f}  "
      f"({horizon_weighted_loss.raw_weights.sum()/HORIZON:.2f}x skala MAE polos)")
print(f"Setelah normalisasi : rata-rata = {horizon_weighted_loss.weights_np.mean():.4f}  "
      f"(rentang {horizon_weighted_loss.weights_np.min():.3f} - {horizon_weighted_loss.weights_np.max():.3f})")

# --- Sanity check: pada bobot uniform, loss berbobot HARUS identik dengan MAE polos ---
yt = tf.constant(rng.normal(size=(8, HORIZON, 1)).astype("float32"))
yp = tf.constant(rng.normal(size=(8, HORIZON, 1)).astype("float32"))
uniform = HorizonWeightedMAELoss(HORIZON, w_start=1.0, w_step=0.0, normalize=True)
print(f"\nSANITY CHECK  MAE polos              = {float(custom_mae_loss(yt, yp)):.6f}")
print(f"SANITY CHECK  weighted (w_step=0)    = {float(uniform(yt, yp)):.6f}  -> harus identik")
print(f"SANITY CHECK  weighted (w_step=0.2)  = {float(horizon_weighted_loss(yt, yp)):.6f}  -> berbeda (reweighted)")
assert abs(float(custom_mae_loss(yt, yp)) - float(uniform(yt, yp))) < 1e-6, "Loss berbobot tidak konsisten"
print("ASSERTION LULUS: implementasi loss berbobot konsisten terhadap MAE polos.")

# %% [markdown]
# ---
# # 11. Custom Callbacks — Implementation

# %%
class CustomCallback:

    def __init__(self):
        self.model         = None
        self.optimizer     = None
        self.stop_training = False

    def set_context(self, model, optimizer):
        self.model, self.optimizer = model, optimizer

    def on_train_begin(self, logs=None):  pass
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None):   pass
    def on_train_end(self, logs=None):    pass


class CustomEarlyStopping(CustomCallback):

    def __init__(self, monitor="val_loss", patience=5, min_delta=1e-6,
                 restore_best_weights=True, verbose=1):
        super().__init__()
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.restore_best_weights, self.verbose = restore_best_weights, verbose

    def on_train_begin(self, logs=None):
        self.best        = np.inf
        self.wait        = 0
        self.best_epoch  = 0
        self.best_weights = None
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        if current < self.best - self.min_delta:                 
            self.best, self.best_epoch, self.wait = current, epoch, 0
            if self.restore_best_weights:
                self.best_weights = [w.copy() for w in self.model.get_weights()]
        else:                                                    
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f"    [CustomEarlyStopping] Berhenti pada epoch {epoch+1}: "
                          f"{self.monitor} tidak membaik selama {self.patience} epoch. "
                          f"Terbaik = {self.best:.6f} @ epoch {self.best_epoch+1}.")

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.verbose:
                print(f"    [CustomEarlyStopping] Bobot dipulihkan ke epoch terbaik "
                      f"({self.best_epoch+1}, {self.monitor}={self.best:.6f}).")


class CustomReduceLROnPlateau(CustomCallback):

    def __init__(self, monitor="val_loss", factor=0.5, patience=2,
                 min_delta=1e-6, min_lr=1e-6, cooldown=0, verbose=1):
        super().__init__()
        self.monitor, self.factor, self.patience = monitor, factor, patience
        self.min_delta, self.min_lr, self.cooldown, self.verbose = min_delta, min_lr, cooldown, verbose
        self.lr_history = []

    @staticmethod
    def _get_lr(optimizer):
        lr = optimizer.learning_rate
        return float(lr.numpy()) if hasattr(lr, "numpy") else float(lr)

    @staticmethod
    def _set_lr(optimizer, value):
        try:
            optimizer.learning_rate.assign(value)
        except Exception:
            optimizer.learning_rate = value

    def on_train_begin(self, logs=None):
        self.best = np.inf
        self.wait = 0
        self.cooldown_counter = 0
        self.lr_history = []

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        lr_now  = self._get_lr(self.optimizer)
        self.lr_history.append(lr_now)
        (logs or {})["lr"] = lr_now
        if current is None:
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
            return

        if current < self.best - self.min_delta:
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(lr_now * self.factor, self.min_lr)
                if new_lr < lr_now - 1e-12:
                    self._set_lr(self.optimizer, new_lr)
                    if self.verbose:
                        print(f"    [CustomReduceLROnPlateau] {self.monitor} stagnan "
                              f"{self.patience} epoch -> LR {lr_now:.2e} → {new_lr:.2e}")
                self.wait = 0
                self.cooldown_counter = self.cooldown


class CustomPredictionPlotCallback(CustomCallback):

    def __init__(self, enc_sample, dec_sample, y_sample, scaler, every=10, seq2seq=True):
        super().__init__()
        self.enc, self.dec, self.y = enc_sample, dec_sample, y_sample
        self.scaler, self.every, self.seq2seq = scaler, every, seq2seq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return
        inp  = [self.enc, self.dec] if self.seq2seq else self.enc
        pred = self.model(inp, training=False).numpy()
        p = self.scaler.inverse_transform(pred[0].reshape(-1, 1)).ravel()
        a = self.scaler.inverse_transform(self.y[0].reshape(-1, 1)).ravel()
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.plot(range(1, len(a) + 1), a, "o-", color="#2b6cb0", ms=3, label="Data Aktual")
        ax.plot(range(1, len(p) + 1), p, "x--", color="#dd6b20", ms=4, label="Hasil Prediksi")
        ax.set_title(f"[CustomPredictionPlotCallback] Epoch {epoch+1} — "
                     f"val_loss={logs.get('val_loss', float('nan')):.6f}", fontsize=10)
        ax.set_xlabel("Jam ke Depan (Time Step)"); ax.set_ylabel("Nilai Close")
        ax.legend(fontsize=8); plt.tight_layout(); plt.show()

print("Custom callbacks siap: CustomEarlyStopping | CustomReduceLROnPlateau | CustomPredictionPlotCallback")
print("Seluruhnya dibangun di atas kelas dasar CustomCallback yang ditulis sendiri.")

# %% [markdown]
# ---
# # 12. Custom Training Loop

# %%
def train_with_gradient_tape(model, train_ds, val_ds, loss_fn, epochs,
                             optimizer=None, callbacks=None, clip_norm=1.0,
                             model_name="model", verbose=1):

    optimizer = optimizer or keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    callbacks = callbacks or []
    for cb in callbacks:
        cb.set_context(model, optimizer)

    train_loss_metric = keras.metrics.Mean(name="train_loss")
    val_loss_metric   = keras.metrics.Mean(name="val_loss")
    history = {"loss": [], "val_loss": [], "lr": [], "epoch_time": []}

    # --------------------------- Custom Training ----------------------------
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:                       
            y_pred = model(x_batch, training=True)
            loss   = loss_fn(y_batch, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        if clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)  
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x_batch, y_batch):
        return loss_fn(y_batch, model(x_batch, training=False))

    # ------------------------------------- loop -----------------------------------------
    if verbose:
        print("=" * 84)
        print(f"CUSTOM TRAINING LOOP (tf.GradientTape) — {model_name}")
        print(f"  optimizer={optimizer.__class__.__name__} | lr={LEARNING_RATE} | "
              f"loss={getattr(loss_fn, '__name__', loss_fn.__class__.__name__)} | "
              f"epochs={epochs} | clip_norm={clip_norm}")
        print(f"  trainable params = {int(sum(np.prod(v.shape) for v in model.trainable_variables)):,}")
        print("=" * 84)

    for cb in callbacks:
        cb.on_train_begin()

    for epoch in range(epochs):
        t0 = time.time()
        for cb in callbacks:
            cb.on_epoch_begin(epoch)

        train_loss_metric.reset_state()
        val_loss_metric.reset_state()

        for x_batch, y_batch in train_ds:                       # ---- fase latih ----
            train_loss_metric.update_state(train_step(x_batch, y_batch))
        for x_batch, y_batch in val_ds:                         # ---- fase validasi ----
            val_loss_metric.update_state(val_step(x_batch, y_batch))

        loss_v = float(train_loss_metric.result())
        vloss_v = float(val_loss_metric.result())
        lr_v = CustomReduceLROnPlateau._get_lr(optimizer)
        dt = time.time() - t0

        history["loss"].append(loss_v); history["val_loss"].append(vloss_v)
        history["lr"].append(lr_v);     history["epoch_time"].append(dt)

        logs = {"loss": loss_v, "val_loss": vloss_v, "lr": lr_v}

        if verbose:                                             # jumlah epoch, loss, val_loss
            print(f"Epoch {epoch+1:3d}/{epochs} - {dt:5.1f}s - loss: {loss_v:.6f} - "
                  f"val_loss: {vloss_v:.6f} - lr: {lr_v:.2e}")

        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)
        if any(cb.stop_training for cb in callbacks):
            break

    for cb in callbacks:
        cb.on_train_end()

    if verbose:
        print(f"Selesai — {len(history['loss'])} epoch | "
              f"val_loss terbaik = {min(history['val_loss']):.6f} | "
              f"total {sum(history['epoch_time']):.0f}s\n")
    return history


print("Custom training loop siap. Seluruh model dilatih dengan fungsi ini (tanpa model.fit).")

# %% [markdown] 
# ---
# # 13. Model Architecture
# 
# ## 13.1 Teacher Forcing

# %%
def build_baseline_lstm(name="model_baseline_LSTM", units=LSTM_UNITS,
                        attention=None, custom_blocks=False):

    enc_in = keras.Input(shape=(WINDOW_SIZE, N_FEATURES), name="encoder_input")

    x = layers.LSTM(units, return_sequences=True, name="lstm_1")(enc_in)

    if attention == "keras":
        attn = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM,
                                         name="mha_self")(query=x, value=x, key=x)
        x = layers.Add(name="attn_residual")([x, attn])
        x = layers.LayerNormalization(name="attn_norm")(x)
    elif attention == "custom":
        attn = CustomMultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM,
                                        name="custom_mha_self")([x, x])
        x = layers.Add(name="attn_residual")([x, attn])
        x = CustomLayerNormalization(name="custom_attn_norm")(x)

    x = layers.LSTM(units, return_sequences=False, name="lstm_2")(x)

    if custom_blocks:
        x = CustomDense(units * 2, name="custom_dense_hidden")(x)
        x = CustomLeakyReLU(alpha=0.1, name="custom_leaky_relu")(x)
        x = CustomDropout(rate=0.15, name="custom_dropout")(x)
    else:
        x = CustomDense(units * 2, activation="relu", name="custom_dense_hidden")(x)

    delta = CustomDense(HORIZON, name="custom_dense_delta_head")(x)      # (B, 24)
    delta = layers.Reshape((HORIZON, 1), name="reshape_horizon")(delta)  # (B, 24, 1)
    out   = LevelAnchorAdd(target_idx=TARGET_COL_IDX, name="level_anchor")([delta, enc_in])

    return keras.Model(inputs=enc_in, outputs=out, name=name)


def build_seq2seq_lstm(name="model_seq2seq_LSTM", units=LSTM_UNITS,
                       attention=None, custom_blocks=False):

    enc_in = keras.Input(shape=(WINDOW_SIZE, N_FEATURES), name="encoder_input")
    dec_in = keras.Input(shape=(HORIZON, 1),              name="decoder_input")

    # ---------------- ENCODER ----------------
    enc_seq, state_h, state_c = layers.LSTM(
        units, return_sequences=True, return_state=True, name="encoder_lstm")(enc_in)

    # ---------------- DECODER ----------------
    dec_seq = layers.LSTM(units, return_sequences=True, name="decoder_lstm")(
        dec_in, initial_state=[state_h, state_c])

    # ---------------- CROSS-ATTENTION ----------------
    if attention == "keras":
        ctx = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM,
                                        name="mha_cross")(query=dec_seq, value=enc_seq, key=enc_seq)
        dec_seq = layers.Add(name="attn_residual")([dec_seq, ctx])
        dec_seq = layers.LayerNormalization(name="attn_norm")(dec_seq)
    elif attention == "custom":
        ctx = CustomMultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM,
                                       name="custom_mha_cross")([dec_seq, enc_seq])
        dec_seq = layers.Add(name="attn_residual")([dec_seq, ctx])
        dec_seq = CustomLayerNormalization(name="custom_attn_norm")(dec_seq)

    # ---------------- OUTPUT ----------------
    if custom_blocks:
        z = CustomDense(units, name="custom_dense_hidden")(dec_seq)
        z = CustomLeakyReLU(alpha=0.1, name="custom_leaky_relu")(z)
        z = CustomDropout(rate=0.15, name="custom_dropout")(z)
    else:
        z = CustomDense(units, activation="relu", name="custom_dense_hidden")(dec_seq)

    delta = CustomDense(1, name="custom_dense_delta_head")(z)            # (B, 24, 1)
    out   = LevelAnchorAdd(target_idx=TARGET_COL_IDX, name="level_anchor")([delta, enc_in])

    return keras.Model(inputs=[enc_in, dec_in], outputs=out, name=name)


print("Builder Functional API siap: build_baseline_lstm() | build_seq2seq_lstm()")

# %% [markdown]
# ## 13.2 Seq2Seq via *Model Subclassing*

# %%
@keras.saving.register_keras_serializable(package="DLTM")
class Seq2SeqLSTMSubclassed(keras.Model):

    def __init__(self, window_size, n_features, horizon, units=LSTM_UNITS,
                 target_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.window_size, self.n_features = window_size, n_features
        self.horizon, self.units, self.target_idx = horizon, units, target_idx

        self.encoder_lstm = layers.LSTM(units, return_sequences=True,
                                        return_state=True, name="encoder_lstm")
        self.decoder_lstm = layers.LSTM(units, return_sequences=True,
                                        return_state=True, name="decoder_lstm")
        self.hidden_dense = CustomDense(units, activation="relu", name="custom_dense_hidden")
        self.delta_head   = CustomDense(1, name="custom_dense_delta_head")
        self.anchor       = LevelAnchorAdd(target_idx=target_idx, name="level_anchor")

    def call(self, inputs, training=False):
        enc_in, dec_in = inputs[0], inputs[1]
        _, state_h, state_c = self.encoder_lstm(enc_in, training=training)
        dec_seq, _, _ = self.decoder_lstm(dec_in, initial_state=[state_h, state_c],
                                          training=training)
        delta = self.delta_head(self.hidden_dense(dec_seq))
        return self.anchor([delta, enc_in])

    def predict_autoregressive(self, enc_in):

        enc_in = tf.convert_to_tensor(enc_in, dtype=tf.float32)
        _, state_h, state_c = self.encoder_lstm(enc_in, training=False)
        anchor = enc_in[:, -1:, self.target_idx : self.target_idx + 1]   # (B, 1, 1)
        step_in, preds = anchor, []
        for _ in range(self.horizon):
            dec_out, state_h, state_c = self.decoder_lstm(
                step_in, initial_state=[state_h, state_c], training=False)
            delta = self.delta_head(self.hidden_dense(dec_out))          # (B, 1, 1)
            y_hat = delta + anchor
            preds.append(y_hat)
            step_in = y_hat                                              # umpan balik prediksi
        return tf.concat(preds, axis=1)                                  # (B, H, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"window_size": self.window_size, "n_features": self.n_features,
                    "horizon": self.horizon, "units": self.units,
                    "target_idx": self.target_idx})
        return cfg


print("Builder Model Subclassing siap: Seq2SeqLSTMSubclassed")

# %% [markdown]
# ## 13.3 Architecture Contract Verification

# %%
keras.utils.set_random_seed(SEED)

model_baseline_LSTM        = build_baseline_lstm("model_baseline_LSTM")
model_seq2seq_LSTM         = build_seq2seq_lstm("model_seq2seq_LSTM")

model_seq2seq_subclassed   = Seq2SeqLSTMSubclassed(
    window_size=WINDOW_SIZE, n_features=N_FEATURES, horizon=HORIZON,
    units=LSTM_UNITS, target_idx=TARGET_COL_IDX, name="model_seq2seq_subclassed")

model_baseline_LSTM_mha    = build_baseline_lstm("model_baseline_LSTM_mha", attention="keras")
model_seq2seq_LSTM_mha     = build_seq2seq_lstm("model_seq2seq_LSTM_mha",  attention="keras")

model_baseline_LSTM_custom = build_baseline_lstm("model_baseline_LSTM_custom",
                                                 attention="custom", custom_blocks=True)
best_model_seq2seq_LSTM    = build_seq2seq_lstm("best_model_seq2seq_LSTM",
                                                attention="custom", custom_blocks=True)

ALL_MODELS = [
    ("model_baseline_LSTM",        model_baseline_LSTM,        False),
    ("model_seq2seq_LSTM",         model_seq2seq_LSTM,         True),
    ("model_seq2seq_subclassed",   model_seq2seq_subclassed,   True),
    ("model_baseline_LSTM_mha",    model_baseline_LSTM_mha,    False),
    ("model_seq2seq_LSTM_mha",     model_seq2seq_LSTM_mha,     True),
    ("model_baseline_LSTM_custom", model_baseline_LSTM_custom, False),
    ("best_model_seq2seq_LSTM",    best_model_seq2seq_LSTM,    True),
]

# --------------------------- VERIFICATION ---------------------------
print("=" * 30)
print("VERIFIKASI KONTRAK ARSITEKTUR")
print("=" * 30)
rows = []
for nm, mdl, is_s2s in ALL_MODELS:
    inp = [enc_train[:4], dec_train[:4]] if is_s2s else enc_train[:4]
    out = mdl(inp, training=False)
    n_params = int(sum(np.prod(v.shape) for v in mdl.trainable_variables))
    custom_layer_names = [l.__class__.__name__ for l in getattr(mdl, "layers", [])
                          if l.__class__.__name__.startswith("Custom") or
                             l.__class__.__name__ == "LevelAnchorAdd"]
    if not custom_layer_names:   # model subclassed
        custom_layer_names = ["CustomDense", "CustomDense", "LevelAnchorAdd"]
    rows.append({
        "Model": nm,
        "Output shape": str(tuple(out.shape)),
        "Steps": int(out.shape[1]),
        "24-step?": "YA" if int(out.shape[1]) == HORIZON else "TIDAK",
        "Params": n_params,
        "Custom layers": len(custom_layer_names),
    })
    assert tuple(out.shape)[1:] == (HORIZON, 1), f"{nm}: bentuk keluaran salah -> {out.shape}"

display(pd.DataFrame(rows).set_index("Model").style.format({"Params": "{:,}"})
        .set_caption("Seluruh model menghasilkan tepat 24 langkah -> Kriteria 2 (24-step) terpenuhi"))
print("ASSERTION LULUS: seluruh model menghasilkan (batch, 24, 1) — multi-step 24 langkah,")
print("                 bukan single-step dan bukan < 24 langkah.")

# %%
print("=" * 42)
print("RINGKASAN ARSITEKTUR — model_seq2seq_LSTM")
print("=" * 42)
model_seq2seq_LSTM.summary(line_length=100)

# %%
print("=" * 47)
print("RINGKASAN ARSITEKTUR — best_model_seq2seq_LSTM")
print("=" * 47)
best_model_seq2seq_LSTM.summary(line_length=100)

# --- Custom Layer Inventory ---
print("\nINVENTARIS CUSTOM LAYER pada best_model_seq2seq_LSTM:")
for l in best_model_seq2seq_LSTM.layers:
    cls = l.__class__.__name__
    if cls.startswith("Custom") or cls == "LevelAnchorAdd":
        print(f"  - {l.name:<28s} : {cls}")

# %%
try:
    display(keras.utils.plot_model(best_model_seq2seq_LSTM, show_shapes=True,
                                   show_layer_names=True, dpi=64,
                                   rankdir="TB", expand_nested=False))
except Exception as e:
    print(f"(plot_model dilewati: {type(e).__name__} — memerlukan graphviz/pydot)")

# %% [markdown]
# ---
# # 14. Training Execution  
# 
# ## 14.1 Model 1 — `model_baseline_LSTM`

# %%
keras.utils.set_random_seed(SEED)
histories = {}

es_cb = CustomEarlyStopping(monitor="val_loss", patience=PATIENCE_ES, restore_best_weights=True)
lr_cb = CustomReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_LR, min_lr=1e-6)

histories["model_baseline_LSTM"] = train_with_gradient_tape(
    model=model_baseline_LSTM,
    train_ds=train_ds_base, val_ds=val_ds_base,
    loss_fn=custom_mae_loss,                       
    epochs=EPOCHS_MAIN,
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    callbacks=[es_cb, lr_cb],                      
    model_name="model_baseline_LSTM (Baseline LSTM + CustomDense)",
)

# %% [markdown]
# ## 14.2 Model 2 — `model_seq2seq_LSTM`

# %%
keras.utils.set_random_seed(SEED)

histories["model_seq2seq_LSTM"] = train_with_gradient_tape(
    model=model_seq2seq_LSTM,
    train_ds=train_ds_s2s, val_ds=val_ds_s2s,
    loss_fn=custom_mae_loss,
    epochs=EPOCHS_MAIN,
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    callbacks=[CustomEarlyStopping(patience=PATIENCE_ES, restore_best_weights=True),
               CustomReduceLROnPlateau(factor=0.5, patience=PATIENCE_LR, min_lr=1e-6)],
    model_name="model_seq2seq_LSTM (Seq2Seq + Teacher Forcing, Functional API)",
)

# %% [markdown]
# ## 14.3 Model 3 — `model_seq2seq_subclassed`

# %%
keras.utils.set_random_seed(SEED)

histories["model_seq2seq_subclassed"] = train_with_gradient_tape(
    model=model_seq2seq_subclassed,
    train_ds=train_ds_s2s, val_ds=val_ds_s2s,
    loss_fn=custom_mae_loss,
    epochs=EPOCHS_DEMO,
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    callbacks=[CustomEarlyStopping(patience=PATIENCE_ES, restore_best_weights=True),
               CustomReduceLROnPlateau(factor=0.5, patience=PATIENCE_LR, min_lr=1e-6)],
    model_name="model_seq2seq_subclassed (Seq2Seq + Teacher Forcing, Model Subclassing)",
)

# %% [markdown]
# ## 14.4 Model 4 — Add **Multi-Head Attention**

# %%
keras.utils.set_random_seed(SEED)

histories["model_seq2seq_LSTM_mha"] = train_with_gradient_tape(
    model=model_seq2seq_LSTM_mha,
    train_ds=train_ds_s2s, val_ds=val_ds_s2s,
    loss_fn=custom_mae_loss, epochs=EPOCHS_DEMO,
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    callbacks=[CustomEarlyStopping(patience=PATIENCE_ES, restore_best_weights=True),
               CustomReduceLROnPlateau(factor=0.5, patience=PATIENCE_LR, min_lr=1e-6)],
    model_name="model_seq2seq_LSTM_mha (Seq2Seq + Keras MultiHeadAttention cross-attn)",
)

# %% [markdown]
# ## 14.5 Model 5 — `model_baseline_LSTM_custom`

# %%
keras.utils.set_random_seed(SEED)

histories["model_baseline_LSTM_custom"] = train_with_gradient_tape(
    model=model_baseline_LSTM_custom,
    train_ds=train_ds_base, val_ds=val_ds_base,
    loss_fn=custom_mae_loss, epochs=EPOCHS_DEMO,
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    callbacks=[CustomEarlyStopping(patience=PATIENCE_ES, restore_best_weights=True),
               CustomReduceLROnPlateau(factor=0.5, patience=PATIENCE_LR, min_lr=1e-6)],
    model_name="model_baseline_LSTM_custom (Baseline + CustomMHA + Custom Dropout/Norm/Activation)",
)

# %% [markdown]
# ## 14.6 Model 6 — `best_model_seq2seq_LSTM`
# 

# %%
keras.utils.set_random_seed(SEED)

plot_cb = CustomPredictionPlotCallback(
    enc_sample=enc_val[:1], dec_sample=dec_val[:1], y_sample=y_val[:1],
    scaler=target_scaler, every=max(EPOCHS_MAIN // 2, 1), seq2seq=True)

es_best = CustomEarlyStopping(monitor="val_loss", patience=PATIENCE_ES, restore_best_weights=True)
lr_best = CustomReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=PATIENCE_LR, min_lr=1e-6)

histories["best_model_seq2seq_LSTM"] = train_with_gradient_tape(
    model=best_model_seq2seq_LSTM,
    train_ds=train_ds_s2s, val_ds=val_ds_s2s,
    loss_fn=horizon_weighted_loss,                 
    epochs=EPOCHS_MAIN,
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    callbacks=[es_best, lr_best, plot_cb],
    model_name="best_model_seq2seq_LSTM (Seq2Seq + CustomMHA + Horizon-Weighted Loss)",
)

# %% [markdown]
# ## 14.7 Learning Curve and Learning Rate

# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, (nm, h) in zip(axes.ravel(), histories.items()):
    ep = np.arange(1, len(h["loss"]) + 1)
    ax.plot(ep, h["loss"],     "o-", ms=2.5, lw=1.2, color="#2b6cb0", label="loss (train)")
    ax.plot(ep, h["val_loss"], "s-", ms=2.5, lw=1.2, color="#c53030", label="val_loss")
    ax.set_title(nm, fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("Loss", fontsize=8)
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.tick_params(labelsize=7)
axes.ravel()[-1].axis("off")
fig.suptitle("Kurva Pembelajaran — Seluruh Model (Custom Training Loop, skala log)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# --- Jejak penurunan LR ---
fig, ax = plt.subplots(figsize=(10, 3.2))
h = histories["best_model_seq2seq_LSTM"]
ax.step(np.arange(1, len(h["lr"]) + 1), h["lr"], where="post", color="#6b46c1", lw=1.6)
ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
ax.set_title("Jejak CustomReduceLROnPlateau — best_model_seq2seq_LSTM", fontweight="bold")
plt.tight_layout(); plt.show()

n_drops = int(np.sum(np.diff(h["lr"]) < 0))
print(f"Jumlah penurunan LR yang dipicu CustomReduceLROnPlateau : {n_drops}")
print(f"LR awal -> LR akhir : {h['lr'][0]:.2e} -> {h['lr'][-1]:.2e}")

# %% [markdown]
# ---
# # 15. Inference: 
# ## 15.1  Teacher Forcing vs **Autoregressive**

# %%
def autoregressive_predict(model, enc_array, horizon=HORIZON, target_idx=TARGET_COL_IDX,
                           batch_size=256, verbose=True):
      
    # Graph Compilation
    @tf.function(reduce_retracing=True)
    def _decode_step(enc_t, dec_t):
        return model([enc_t, dec_t], training=False)

    outputs = []
    n_batches = int(np.ceil(len(enc_array) / batch_size))
    for bi, start in enumerate(range(0, len(enc_array), batch_size)):
        enc_b = enc_array[start : start + batch_size]
        b     = len(enc_b)
        enc_t = tf.convert_to_tensor(enc_b, dtype=tf.float32)

        dec_state = np.zeros((b, horizon, 1), dtype="float32")
        dec_state[:, 0, 0] = enc_b[:, -1, target_idx]      # anchor = observasi terakhir
        pred_out = np.zeros((b, horizon, 1), dtype="float32")

        for h in range(horizon):
            p = _decode_step(enc_t, tf.convert_to_tensor(dec_state)).numpy()
            pred_out[:, h, 0] = p[:, h, 0]
            if h + 1 < horizon:
                dec_state[:, h + 1, 0] = p[:, h, 0]        # umpan balik PREDIKSI
        outputs.append(pred_out)
        if verbose and (bi + 1) % 10 == 0:
            print(f"    batch {bi+1}/{n_batches} selesai", end="\r")
    if verbose:
        print(" " * 40, end="\r")
    return np.concatenate(outputs, axis=0)


print("Fungsi inferensi autoregressive siap.")

# %%
# ------------------------- INFERENCE ON TEST DATA -------------------------
print("=" * 24)
print("INFERENCE ON TEST DATA")
print("=" * 24)

predictions = {}

# --- Model baseline: 24-step prediction ---
for nm, mdl in [("model_baseline_LSTM",        model_baseline_LSTM),
                ("model_baseline_LSTM_mha",    model_baseline_LSTM_mha),
                ("model_baseline_LSTM_custom", model_baseline_LSTM_custom)]:
    t0 = time.time()
    predictions[nm] = mdl.predict(enc_test, batch_size=256, verbose=0)
    print(f"{nm:<30s} | direct multi-step (24 langkah sekaligus) | {time.time()-t0:5.1f}s")

# --- Model Seq2Seq: autoregressive ---
for nm, mdl in [("model_seq2seq_LSTM",      model_seq2seq_LSTM),
                ("model_seq2seq_LSTM_mha",  model_seq2seq_LSTM_mha),
                ("best_model_seq2seq_LSTM", best_model_seq2seq_LSTM)]:
    t0 = time.time()
    predictions[nm] = autoregressive_predict(mdl, enc_test, verbose=False)
    print(f"{nm:<30s} | AUTOREGRESSIVE (iteratif, O(H^2))        | {time.time()-t0:5.1f}s")

# --- Model subclassed: autoregressive ---
t0 = time.time()
sub_preds = []
for s in range(0, len(enc_test), 256):
    sub_preds.append(model_seq2seq_subclassed.predict_autoregressive(enc_test[s : s + 256]).numpy())
predictions["model_seq2seq_subclassed"] = np.concatenate(sub_preds, axis=0)
print(f"{'model_seq2seq_subclassed':<30s} | AUTOREGRESSIVE (state-passing, O(H))     | {time.time()-t0:5.1f}s")

print(f"\nSeluruh prediksi berbentuk {predictions['best_model_seq2seq_LSTM'].shape} "
      f"= (n_sampel_test, 24 langkah, 1)")

# %% [markdown]
# ## 15.2 Evidence of Exposure Bias

# %%
tf_pred = best_model_seq2seq_LSTM.predict([enc_test, dec_test], batch_size=256, verbose=0)
ar_pred = predictions["best_model_seq2seq_LSTM"]

mae_tf = float(np.mean(np.abs(tf_pred - y_test)))
mae_ar = float(np.mean(np.abs(ar_pred - y_test)))

bias_tbl = pd.DataFrame([
    {"Skema Dekode": "Teacher forcing (TIDAK SAH untuk evaluasi)",
     "MAE (scaled)": mae_tf, "MAE (USD)": mae_tf * span_usd,
     "Keterangan": "Decoder diberi nilai SEBENARNYA -> optimistis"},
    {"Skema Dekode": "Autoregressive (SAH — diwajibkan kriteria)",
     "MAE (scaled)": mae_ar, "MAE (USD)": mae_ar * span_usd,
     "Keterangan": "Decoder diberi PREDIKSINYA SENDIRI -> realistis"},
])
display(bias_tbl.style.format({"MAE (scaled)": "{:.6f}", "MAE (USD)": "{:,.0f}"})
        .set_caption("Kuantifikasi exposure bias pada best_model_seq2seq_LSTM"))
print(f"Exposure bias = {mae_ar - mae_tf:+.6f} scaled MAE "
      f"({(mae_ar/mae_tf - 1)*100:+.1f}% terhadap metrik teacher-forced)")
print("\nSeluruh angka yang dilaporkan pada Bagian 16 memakai dekode AUTOREGRESSIVE.")

# %% [markdown]
# ## 15.3 Cross Validation

# %%
sub_ar_oh   = model_seq2seq_subclassed.predict_autoregressive(enc_test[:256]).numpy()  # O(H)
sub_ar_oh2  = autoregressive_predict(model_seq2seq_subclassed, enc_test[:256], verbose=False)  # O(H^2)
max_diff = float(np.max(np.abs(sub_ar_oh - sub_ar_oh2)))
print(f"Selisih maksimum antara dekode O(H) (state-passing) dan O(H^2) (iteratif) : {max_diff:.2e}")
print(f"-> {'SETARA (implementasi generik tervalidasi)' if max_diff < 1e-4 else 'BERBEDA — perlu diperiksa'}")

# %% [markdown]
# ---
# # 16. Evaluation on Test Data

# %%
def evaluate_model(y_true, y_pred, name, span, scaler):
    """Menghitung MAE (metrik utama), RMSE, dan MAPE pada ruang scaled maupun USD."""
    mae_s  = float(np.mean(np.abs(y_true - y_pred)))
    rmse_s = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    yt_usd = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
    yp_usd = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    mae_u  = float(np.mean(np.abs(yt_usd - yp_usd)))
    rmse_u = float(np.sqrt(np.mean((yt_usd - yp_usd) ** 2)))
    mape   = float(np.mean(np.abs((yt_usd - yp_usd) / yt_usd)) * 100)
    return {"Model": name, "MAE (scaled)": mae_s, "RMSE (scaled)": rmse_s,
            "MAE (USD)": mae_u, "RMSE (USD)": rmse_u, "MAPE (%)": mape,
            "Lolos gate <0.015": "LULUS" if mae_s < MAE_GATE_SCALED else "GAGAL"}


results = [evaluate_model(y_test, predictions[nm], nm, span_usd, target_scaler)
           for nm, _, _ in ALL_MODELS]

# --- Benchmark persistence pada pengaturan window test ---
persist_pred = np.repeat(enc_test[:, -1, TARGET_COL_IDX].reshape(-1, 1, 1), HORIZON, axis=1)
results.append(evaluate_model(y_test, persist_pred, "[benchmark] Persistence (random-walk)",
                              span_usd, target_scaler))

res_df = pd.DataFrame(results).set_index("Model").sort_values("MAE (scaled)")
display(res_df.style
        .format({"MAE (scaled)": "{:.6f}", "RMSE (scaled)": "{:.6f}",
                 "MAE (USD)": "{:,.2f}", "RMSE (USD)": "{:,.2f}", "MAPE (%)": "{:.3f}"})
        .set_caption("Evaluasi pada DATA TEST — MAE adalah metrik utama (Ketentuan Penolakan #7)"))

# %%
# =============== VERIFICATION GATE ===============
best_mae_scaled = float(np.mean(np.abs(predictions["best_model_seq2seq_LSTM"] - y_test)))
passed = best_mae_scaled < MAE_GATE_SCALED

print("=" * 16)
print("VERIFIKASI GATE")
print("=" * 16)
print(f"Model                              : best_model_seq2seq_LSTM (Custom Seq2Seq LSTM)")
print(f"Skema dekode                       : AUTOREGRESSIVE")
print(f"MAE data test (SEBELUM inverse scaled) : {best_mae_scaled:.6f}")
print(f"Gate kriteria                      : < {MAE_GATE_SCALED}")
print(f"Margin                             : {MAE_GATE_SCALED - best_mae_scaled:+.6f} "
      f"({(1 - best_mae_scaled/MAE_GATE_SCALED)*100:+.1f}% di bawah gate)")
print(f"STATUS                             : {'*** LULUS ***' if passed else '*** GAGAL ***'}")
print("-" * 84)
print(f"Setara dalam USD                   : {best_mae_scaled * span_usd:,.2f} USD")
print(f"Benchmark persistence              : {mae_persistence:.6f} scaled")
print(f"Rasio model / persistence          : {best_mae_scaled / mae_persistence:.3f}x "
      f"({'mengalahkan' if best_mae_scaled < mae_persistence else 'belum mengalahkan'} random-walk)")
print("=" * 84)

if not passed:
    print("\nDIAGNOSIS bila gate belum terlampaui — coba secara berurutan:")
    print("  1. Naikkan EPOCHS_MAIN (mis. 40) dan PATIENCE_ES (mis. 8).")
    print("  2. Naikkan LSTM_UNITS ke 96/128.")
    print("  3. Turunkan LEARNING_RATE ke 5e-4 agar konvergensi lebih halus.")
    print("  4. Setel horizon_weighted_loss dengan w_step lebih kecil (mis. 0.1).")

# %%
# --- Performance comparison visualization ---
plot_df = res_df.drop(index="[benchmark] Persistence (random-walk)")

fig, axes = plt.subplots(1, 2, figsize=(16, 4.6))

colors = ["#2f855a" if v < MAE_GATE_SCALED else "#c53030" for v in plot_df["MAE (scaled)"]]
axes[0].barh(plot_df.index, plot_df["MAE (scaled)"], color=colors)
axes[0].axvline(MAE_GATE_SCALED, color="black", ls="--", lw=1.6,
                label=f"Gate kriteria = {MAE_GATE_SCALED}")
axes[0].axvline(mae_persistence, color="#6b46c1", ls=":", lw=1.6,
                label=f"Persistence = {mae_persistence:.5f}")
axes[0].set_xlabel("MAE pada data test (scaled)")
axes[0].set_title("MAE Test — Seluruh Model vs Gate Kriteria", fontweight="bold")
axes[0].legend(fontsize=8); axes[0].tick_params(labelsize=8)

# --- Error degradation across the horizon ---
for nm in ["model_baseline_LSTM", "model_seq2seq_LSTM", "best_model_seq2seq_LSTM"]:
    per_step = np.mean(np.abs(predictions[nm] - y_test), axis=(0, 2))
    axes[1].plot(np.arange(1, HORIZON + 1), per_step, "o-", ms=3, lw=1.3, label=nm)
per_step_p = np.mean(np.abs(persist_pred - y_test), axis=(0, 2))
axes[1].plot(np.arange(1, HORIZON + 1), per_step_p, "--", color="#6b46c1", lw=1.3,
             label="[benchmark] persistence")
axes[1].set_xlabel("Langkah horizon (jam ke depan)"); axes[1].set_ylabel("MAE (scaled)")
axes[1].set_title("Degradasi Galat Sepanjang Horizon_24 Langkah", fontweight="bold")
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

print("Interpretasi kurva kanan: galat meningkat secara monoton terhadap horizon — persis")
print("fenomena yang memotivasi HorizonWeightedMAELoss (Kriteria 3 - Advanced), yang memberi")
print("bobot lebih besar pada langkah jauh agar tidak diabaikan optimizer.")

# %% [markdown]
# ## 16.1 Inference and Predicted Line Chart

# %%
# ============ SECOND MODEL INFERENCE ===========
print("=" * 55)
print("KONFIRMASI INFERENCE — KEDUA MODEL (horizon 24 langkah)")
print("=" * 55)
for nm, scheme in [("model_baseline_LSTM", "direct multi-step"),
                   ("model_seq2seq_LSTM",  "AUTOREGRESSIVE")]:
    shp = predictions[nm].shape
    steps = shp[1]
    print(f"  {nm:<26s} | skema: {scheme:<18s} | bentuk = {shp} | "
          f"langkah = {steps} | {'OK (24 langkah penuh)' if steps == HORIZON else 'SALAH'}")
    assert steps == HORIZON, f"{nm}: inference tidak menghasilkan 24 langkah"
print("\nKedua model wajib telah di-inference penuh pada seluruh set test "
      f"({len(y_test):,} jendela x {HORIZON} langkah). Tidak ada pemotongan langkah.")

# %%
SAMPLE_IDX = 0 

# Both models + the best model
INFERENCE_MODELS = [
    ("model_baseline_LSTM",     "Baseline LSTM (direct multi-step)"),
    ("model_seq2seq_LSTM",      "Seq2Seq LSTM (autoregressive)"),
    ("best_model_seq2seq_LSTM", "Best Seq2Seq LSTM (autoregressive)"),
]

def get_pred_actual_usd(model_name, sample_idx):

    y_true_usd = target_scaler.inverse_transform(
        y_test[sample_idx].reshape(-1, 1)).ravel()
    y_pred_usd = target_scaler.inverse_transform(
        predictions[model_name][sample_idx].reshape(-1, 1)).ravel()
    return y_true_usd, y_pred_usd

def plot_prediction_linechart(sample_idx, model_name, title=None, ax=None):

    y_true_usd, y_pred_usd = get_pred_actual_usd(model_name, sample_idx)
    steps = np.arange(0, HORIZON)                     
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(steps, y_true_usd, "o-", color="#1f77b4", ms=4, lw=1.5, label="Data Aktual")
    ax.plot(steps, y_pred_usd, "x--", color="#ff7f0e", ms=5, lw=1.5, label="Hasil Prediksi")
    ax.set_title(f"{title or model_name}\nPerbandingan Prediksi vs Aktual untuk 24 Jam ke Depan "
                 , fontsize=10, fontweight="bold")
    ax.set_xlabel("Jam ke Depan (Time Step)"); ax.set_ylabel("Nilai Close")
    ax.set_xticks(np.arange(0, HORIZON, 2))
    ax.legend(fontsize=8)
    if standalone:
        plt.tight_layout(); plt.show()
    return y_true_usd, y_pred_usd


# ---- Line chart for each model ----
for nm, title in INFERENCE_MODELS:
    plot_prediction_linechart(SAMPLE_IDX, nm, title=title)

# %%
# ---- Overlay head-to-head: verification of performance ----
y_true_usd, base_usd = get_pred_actual_usd("model_baseline_LSTM", SAMPLE_IDX)
_,          s2s_usd  = get_pred_actual_usd("model_seq2seq_LSTM",  SAMPLE_IDX)
steps = np.arange(0, HORIZON)

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(steps, y_true_usd, "o-",  color="#1f77b4", ms=5, lw=2.0, label="Data Aktual")
ax.plot(steps, base_usd,   "x--", color="#ff7f0e", ms=6, lw=1.6, label="model_baseline_LSTM (prediksi)")
ax.plot(steps, s2s_usd,    "s--", color="#2f855a", ms=5, lw=1.6, label="model_seq2seq_LSTM (prediksi)")
ax.set_title("Perbandingan Head-to-Head — Baseline vs Seq2Seq untuk 24 Jam ke Depan "
             , fontsize=11, fontweight="bold")
ax.set_xlabel("Jam ke Depan (Time Step)"); ax.set_ylabel("Nilai Close")
ax.set_xticks(np.arange(0, HORIZON))
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# ---- Robustness across samples ----
sample_points = [int(k * (len(y_test) - 1) / 3) for k in range(4)]
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for col, idx in enumerate(sample_points):
    plot_prediction_linechart(idx, "model_baseline_LSTM",
                              title="Baseline LSTM", ax=axes[0, col])
    plot_prediction_linechart(idx, "model_seq2seq_LSTM",
                              title="Seq2Seq LSTM", ax=axes[1, col])
fig.suptitle("24-Step Inference on Several Test Points — "
             "Top Row: Baseline | Bottom Row: Seq2Seq", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 16.2 Comparison Table `model_baseline_LSTM`

# %%
def build_comparison_table(model_name, sample_idx=SAMPLE_IDX):

    y_true_usd, y_pred_usd = get_pred_actual_usd(model_name, sample_idx)
    tbl = pd.DataFrame({
        "Jam ke":         np.arange(1, HORIZON + 1),
        "Data Aktual":    y_true_usd,
        "Hasil Prediksi": y_pred_usd,
        "Selisih":        y_true_usd - y_pred_usd,      
    })
    tbl["Selisih Absolut"] = tbl["Selisih"].abs()
    tbl["Galat (%)"] = (tbl["Selisih Absolut"] / tbl["Data Aktual"]) * 100
    return tbl

def show_comparison_table(model_name, label):
    tbl = build_comparison_table(model_name)
    assert len(tbl) == HORIZON, "Tabel harus memuat 24 langkah penuh"
    print("=" * 52)
    print(f"TABEL PERBANDINGAN 24 LANGKAH — {label}")
    print("=" * 52)
    display(tbl.style
            .format({"Data Aktual": "{:,.2f}", "Hasil Prediksi": "{:,.6f}",
                     "Selisih": "{:,.6f}", "Selisih Absolut": "{:,.2f}", "Galat (%)": "{:.4f}"})
            .background_gradient(subset=["Selisih Absolut"], cmap="Reds")
            .hide(axis="index")
            .set_caption(f"{label} — perbandingan aktual vs prediksi 24 jam ke depan (USD)"))
    print(f"  Jumlah langkah ditampilkan : {len(tbl)} (horizon penuh)")
    print(f"  MAE sampel ini             : {tbl['Selisih Absolut'].mean():,.2f} USD "
          f"({tbl['Galat (%)'].mean():.4f}% dari harga)")
    print(f"  Galat terbesar             : {tbl['Selisih Absolut'].max():,.2f} USD "
          f"@ jam ke-{int(tbl.loc[tbl['Selisih Absolut'].idxmax(), 'Jam ke'])}")
    print(f"  Galat terkecil             : {tbl['Selisih Absolut'].min():,.2f} USD "
          f"@ jam ke-{int(tbl.loc[tbl['Selisih Absolut'].idxmin(), 'Jam ke'])}")
    return tbl

# --- (1) Tabel penuh 24 langkah — model_baseline_LSTM (WAJIB) ---
comparison_baseline = show_comparison_table("model_baseline_LSTM", "model_baseline_LSTM")

# %% [markdown]
# ## 16.3 Comparison Table `model_seq2seq_LSTM`

# %%
# --- Comparison table 24 steps — model_seq2seq_LSTM ---
comparison_seq2seq = show_comparison_table("model_seq2seq_LSTM",
                                           "model_seq2seq_LSTM")


# %% [markdown]
# ## 16.4 Comparison Table `best_model_seq2seq_LSTM`

# %%
# --- Comparison table 24 steps — best_model_seq2seq_LSTM ---
comparison_best = show_comparison_table("best_model_seq2seq_LSTM",
                                        "best_model_seq2seq_LSTM")

# %% [markdown]
# ## 16.5 Comparison Table (Both Models)

# %%
# --- Comparison table 24 steps ---
yt_usd, base_usd = get_pred_actual_usd("model_baseline_LSTM", SAMPLE_IDX)
_,       s2s_usd = get_pred_actual_usd("model_seq2seq_LSTM",  SAMPLE_IDX)

combined = pd.DataFrame({
    "Jam ke":                 np.arange(1, HORIZON + 1),
    "Data Aktual":            yt_usd,
    "Baseline — Prediksi":    base_usd,
    "Baseline — Selisih":     yt_usd - base_usd,
    "Seq2Seq — Prediksi":     s2s_usd,
    "Seq2Seq — Selisih":      yt_usd - s2s_usd,
})
assert len(combined) == HORIZON, "Tabel gabungan harus 24 langkah penuh"

print("=" * 70)
print(f"COMPARISON TABLE 24 STEPS — model_baseline_LSTM vs model_seq2seq_LSTM")
print("=" * 70)
display(combined.style
        .format({"Data Aktual": "{:,.2f}",
                 "Baseline — Prediksi": "{:,.2f}", "Baseline — Selisih": "{:,.2f}",
                 "Seq2Seq — Prediksi": "{:,.2f}", "Seq2Seq — Selisih": "{:,.2f}"})
        .background_gradient(subset=["Baseline — Selisih", "Seq2Seq — Selisih"],
                             cmap="RdBu", vmin=-combined[["Baseline — Selisih",
                             "Seq2Seq — Selisih"]].abs().max().max(),
                             vmax=combined[["Baseline — Selisih",
                             "Seq2Seq — Selisih"]].abs().max().max())
        .hide(axis="index")
        .set_caption("Verifikasi head-to-head: prediksi 24 jam kedua model wajib vs data aktual (USD)"))

mae_base_s = float(np.mean(np.abs(base_usd - yt_usd)))
mae_s2s_s  = float(np.mean(np.abs(s2s_usd - yt_usd)))
print(f"  MAE sampel — model_baseline_LSTM : {mae_base_s:,.2f} USD")
print(f"  MAE sampel — model_seq2seq_LSTM  : {mae_s2s_s:,.2f} USD")
print(f"  Model lebih akurat pada sampel ini: "
      f"{'model_seq2seq_LSTM' if mae_s2s_s < mae_base_s else 'model_baseline_LSTM'}")
print("\nCatatan: MAE agregat lintas SELURUH set test (bukan satu sampel) dilaporkan pada")
print("tabel evaluasi Bagian 16 — itulah metrik utama untuk perbandingan performa kedua model.")

# %% [markdown]
# ## 16.6 Diagnostic of Residuals

# %%
resid = (y_test - predictions["best_model_seq2seq_LSTM"]).ravel() * span_usd

fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))

axes[0].hist(resid, bins=80, color="#2b6cb0", alpha=.85)
axes[0].axvline(0, color="black", ls="--", lw=1)
axes[0].set_title(f"Distribusi Residual (USD)\nmean={resid.mean():,.1f} | std={resid.std():,.1f}",
                  fontweight="bold", fontsize=9)
axes[0].set_xlabel("Residual (aktual − prediksi), USD")

sm.qqplot(resid[::10], line="s", ax=axes[1], markersize=1.5, alpha=.5)
axes[1].set_title("Q-Q Plot Residual vs Normal", fontweight="bold", fontsize=9)

pred_flat = (predictions["best_model_seq2seq_LSTM"].ravel() * span_usd) + target_scaler.data_min_[0]
axes[2].scatter(pred_flat[::20], resid[::20], s=2, alpha=.25, color="#c53030")
axes[2].axhline(0, color="black", ls="--", lw=1)
axes[2].set_xlabel("Nilai prediksi (USD)"); axes[2].set_ylabel("Residual (USD)")
axes[2].set_title("Residual vs Nilai Prediksi", fontweight="bold", fontsize=9)

plt.tight_layout(); plt.show()

print(f"Rata-rata residual (bias)  : {resid.mean():>10,.2f} USD  "
      f"-> {'mendekati nol: model tidak bias sistematis' if abs(resid.mean()) < resid.std()*0.1 else 'terdapat bias sistematis'}")
print(f"Deviasi standar residual   : {resid.std():>10,.2f} USD")
print(f"Skewness                   : {float(pd.Series(resid).skew()):>10.4f}")
print(f"Kurtosis (excess)          : {float(pd.Series(resid).kurt()):>10.4f}  "
      f"-> {'leptokurtic (heavy tails), khas aset kripto' if pd.Series(resid).kurt() > 0 else 'platykurtic'}")
print("\nPlot ketiga menunjukkan heteroskedastisitas: sebaran residual melebar pada level harga")
print("tinggi. Ini konsisten dengan volatility clustering yang terlihat pada dekomposisi (5.4)")
print("dan merupakan batasan model yang diakui secara eksplisit — bukan diabaikan.")

# %%
# ============ TOP-UP VERIFICATION ============
print("=" * 35)
print("VERIFIKASI MUAT-ULANG MODEL .keras")
print("=" * 35)

for fname, (orig_model, is_s2s, _) in MODEL_FILES.items():
    path = os.path.join(SUBMISSION_DIR, fname)
    reloaded = keras.models.load_model(path)
    inp = [enc_test[:64], dec_test[:64]] if is_s2s else enc_test[:64]
    p_orig = orig_model(inp, training=False).numpy()
    p_load = reloaded(inp, training=False).numpy()
    max_diff = float(np.max(np.abs(p_orig - p_load)))
    ok = max_diff < 1e-6
    print(f"  {fname:<32s} | max |Δ prediksi| = {max_diff:.2e} | "
          f"{'LULUS — bit-identik' if ok else 'GAGAL'}")
    assert ok, f"{fname}: model hasil muat-ulang tidak identik dengan model asli"

print("\nASSERTION LULUS: seluruh model .keras dapat dimuat ulang tanpa argumen custom_objects")
print("dan menghasilkan prediksi yang identik. Artefak submission valid.")


