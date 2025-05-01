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

## 📊 Evaluation

### Questions for Video-Specific Evaluation

For each video, the ergonomists answered the following questions:

1. **How would you rate the quality of the system’s recommendation?**  
   (1 = Very poor | 2 = Poor | 3 = Fair | 4 = Good | 5 = Excellent)

2. **How effective do you think the recommendation would be at reducing the risk of injury?**  
   (1 = Not at all effective | 2 = Slightly effective | 3 = Moderately effective | 4 = Very effective | 5 = Extremely effective)

3. **How clear was the recommendation in terms of what specific action should be taken?**  
   (1 = Very unclear | 2 = Unclear | 3 = Neutral | 4 = Clear | 5 = Very clear)

4. **How appropriate was the system’s recommendation for addressing the observed posture issue?**  
   (1 = Very inappropriate | 2 = Inappropriate | 3 = Neutral | 4 = Appropriate | 5 = Very appropriate)

---

### Questions for Overall System Evaluation

At the end of the evaluation, the ergonomists answered the following questions to assess the overall system:

1. **Do you find the system’s recommendations to be adaptive to different types of posture problems?**  
   (1 = Very repetitive | 2 = Somewhat repetitive | 3 = Neutral | 4 = Diverse | 5 = Highly adaptive)

2. **Is the information provided alongside the recommendation sufficient for the user to understand how to improve their posture?**  
   (1 = Not sufficient at all | 2 = Slightly sufficient | 3 = Moderately sufficient | 4 = Very sufficient | 5 = Completely sufficient)

3. **After seeing the feedback, do you think the system helped explain why the initial posture was risky in terms of injury prevention or discomfort?**  
   (1 = Not at all | 2 = Slightly | 3 = Moderately | 4 = Very | 5 = Completely)

4. **Do you think this system could be effectively integrated into a real-world training or workplace environment to address posture-related issues?**  
   (1 = Not useful at all | 2 = Slightly useful | 3 = Moderately useful | 4 = Very useful | 5 = Extremely useful)

5. **Do you think this system helps identify posture-related problems that are otherwise difficult to explain or understand for users?**  
   (1 = Strongly disagree | 2 = Disagree | 3 = Neutral | 4 = Agree | 5 = Strongly agree)

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

---

## 📁 Project Structure (Optional)

You may also include a short breakdown of key files/folders:

```
main.py                      # Main script to run experiments
archives_data/               # Folder for dataset archives
requirements-python3_11.txt  # Environment dependencies
...
```

---

## 📄 License

Include your preferred license here (e.g., MIT, Apache-2.0, etc.)

---

## 📬 Contact

For questions or collaborations, feel free to reach out:

- Your Name – your.email@domain.com  
- [LinkedIn or Website (optional)]

