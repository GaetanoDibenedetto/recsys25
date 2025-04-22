# 🏋️ Lift It Up Right: A Recommender System for Safer Lifting Postures

This repository contains the official implementation of our paper **"Lift It Up Right: A Recommender System for Safer Lifting Postures"**, a system designed to assess and improve lifting posture using ergonomic principles and computer vision.

> ✅ **Tested with Python 3.11**  
> ⚠️ Can be adapted to other Python versions with minor changes (modifications are up to the user)

---

## 📦 Dataset

- The dataset will be **released upon paper acceptance**.
- Once available, place all dataset archives inside the `archives_data/` folder.
- **Important:** All `.zip` files should be extracted in-place (in the same folder where they were downloaded).

---

## ⚙️ Installation

### 1. Clone the repository:

```bash
git clone https://github.com/anonymus/rep_name
cd rep_name
```

### 2. Install dependencies

This project relies on [ViTPose](https://github.com/JunkyByte/easy_ViTPose.git) and [Ultralytics YOLO](https://github.com/ultralytics). All other required packages are listed in:

📄 [`requirements-python3_11.txt`](requirements-python3_11.txt)

Install with:

```bash
pip install -r requirements-python3_11.txt
```

---

## 🚀 Running the System

To reproduce the experiments and run the full pipeline, simply execute:

```bash
python main.py
```

Make sure the dataset is correctly placed and extracted before running.