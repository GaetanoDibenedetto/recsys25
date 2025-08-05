[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# 🏋️ Lift It Up Right: A Recommender System for Safer Lifting Postures

This repository contains the official implementation of our paper **"Lift It Up Right: A Recommender System for Safer Lifting Postures"**, a system designed to assess and improve lifting posture using ergonomic principles and computer vision.

> ✅ **Tested with Python 3.11**  
> ⚠️ Can be adapted to other Python versions with minor changes (modifications are up to the user)

---

## 📦 Dataset

- The dataset has undergone a privacy and ethics review and was approved by TU Delft institutional review board.
- The dataset is published on [ZENODO](https://zenodo.org/records/16743120).
- After downloading, place all dataset archives inside the `archives_data/` folder and extract them in place. After extraction, the folder structure should look like this:
```
└── archives_data/
    ├── annotations.json                
    │
    └── anonymized/
        ├── video_env_1_anonymized/            
        │   ├── gd_0013.mp4
        │   ├── ap_0017.mp4
        │   └── ...
        │
        ├── video_env_1_oblique_anonymized/    
        │   ├── gd_0020.mp4
        │   ├── ap_0013.mp4
        │   └── ...
        │
        ├── video_env_3_anonymized/            
        │   ├── vp_0006_dg.mp4
        │   ├── mp_0003_gd.mp4
        │   └── ...
        │
        └── video_env_3_oblique_anonymized/    
            ├── ...
```
---

## 📚 Background: Ergonomic Principles

Our system is grounded in the **Revised NIOSH Lifting Equation (RNLE)**, a widely accepted standard in ergonomics for evaluating lifting safety.

### Lifting Index (LI)

The **Lifting Index (LI)** is used to assess the risk associated with a lifting task:

$$
LI = \frac{\text{Load Weight}}{\text{RWL}}
$$

Where:
- **Load Weight** is the actual weight lifted.
- **RWL** is the **Recommended Weight Limit**, a value that varies depending on posture.

### Recommended Weight Limit (RWL)

RWL is calculated as:

```
RWL = LC × HM × VM × DM × AM × FM × CM
```

Each multiplier adjusts the baseline weight limit (LC) according to posture.

### RWL Multipliers

| Symbol | Multiplier Description | Formula or Source |
|--------|-------------------------|-------------------|
| `LC`   | Load Constant – the maximum recommended weight under ideal conditions | Based on age and gender (see below) |
| `HM`   | Horizontal Multiplier – based on hand distance from ankles | $$\frac{25}{H}$$ |
| `VM`   | Vertical Multiplier – based on hand height from the floor | $$1 - (0.003 \times \left\lvert V - 75 \right\lvert)$$ |
| `DM`   | Distance Multiplier – based on hand vertical travel distance | $$0.82 + \frac{4.5}{D}$$ |
| `AM`   | Asymmetric Multiplier – based on torso rotation angle | $$1 - (0.0032 \times A)$$ |
| `FM`   | Frequency Multiplier – based on number of lifts per minute | See Table 5 - Chapter 3 in [RNLE guide](https://www.cdc.gov/niosh/docs/94-110/) |
| `CM`   | Coupling Multiplier – based on grip quality |  See Table 7 - Chapter 3 in [RNLE guide](https://www.cdc.gov/niosh/docs/94-110/) |

Where:
- `H` is horizontal reach (cm)
- `V` is vertical height (cm)
- `D` is vertical distance traveled (cm)
- `A` is asymmetry angle (degrees)


#### 🔢 Load Constant (LC)

From the [RNLE guide](https://www.cdc.gov/niosh/docs/94-110/), the default Load Constant is **23 kg**.  
However, we adopt the approach from [ISO 11228](https://www.iso.org/standard/76820.html), which introduces personalization based on the subject’s **age** and **gender**:

```python
if gender == "M":
    LC = 25 if 20 <= age <= 45 else 20
elif gender == "W":
    LC = 20 if 20 <= age <= 45 else 15
```

---
## 📢 Recommendations

Our system provides **visual** and **textual** lifting posture recommendations aimed at reducing the Lifting Index (**LI**) below a safety threshold.

### 🔍 How It Works

* The system analyzes two key frames of the lifting motion: the **start** and **end** of the lift.
* For each frame, it computes three posture-dependent parameters:

  * `H`: horizontal distance of the hands from the ankles
  * `V`: vertical height of the hands from the floor
  * `D`: vertical travel distance between the start and end points

By gradually adjusting the **hand position**, which most strongly influences LI, the system identifies posture corrections that reduce LI below a configurable threshold (`τ = 0.7` for evaluation, assuming a 10 kg load).

### 🧠 Graphical Adjustments

* When the hands move, **surrounding joints**—such as the shoulders, elbows, and hips—are also displaced **proportionally** to maintain anatomical plausibility.

### 📝 Textual Recommendation

Textual feedback is generated based on the difference in joint positions between the original and adjusted postures. A rule-based approach is used to produce readable, body-part-specific instructions:

* If the hands move, the system generates a corresponding textual description.
* If nearby joints like the shoulders, elbows, or hips shift, these are also mentioned.
* For the knees, the system computes the change in joint angle between the original and adjusted poses and describes the change in flexion.

This structured feedback helps users understand not only what to change, but also how those changes affect their overall posture.


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
git clone https://github.com/GaetanoDibenedetto/recsys25.git
cd recsys25
```

### 2. Install dependencies

This project relies on [ViTPose](https://github.com/JunkyByte/easy_ViTPose.git) and [Ultralytics YOLO](https://github.com/ultralytics). All other required packages are listed in:

📄 [`requirements-python3_11.txt`](requirements-python3_11.txt)

Install with:

```bash
pip install -r requirements-python3_11.txt
```

## 📬 Contact

For questions, issues, or collaboration opportunities, contact:

- Gaetano Dibenedetto – gaetano.dibenedetto@uniba.it
- [Personal Web Page](https://gaetanodibenedetto.github.io/) - [Linkedin](https://www.linkedin.com/in/gaetano-dibenedetto/)

## Citing Us

BibTeX format:

```
@inproceedings{DBLP:conf/recsys/Dibenedetto25RecSys,
  author       = {Gaetano Dibenedetto and Pasquale Lops and Marco Polignano and Helma Torkamaan},
  title        = {Lift It Up Right: A Recommender System for Safer Lifting Postures},
  booktitle    = {Proceedings of the 19th {ACM} Conference on Recommender Systems, RecSys 2025, Prague, Czech Republic, September 22–26, 2025},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3705328.3759314},
  doi          = {10.1145/3705328.3759314}
}
```


ACM Reference Format:
```
Gaetano Dibenedetto, Pasquale Lops, Marco Polignano, and Helma Torkamaan. 2025. Lift It Up Right: A Recommender System for Safer Lifting Postures. In Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys ’25), September 22–26, 2025, Prague, Czech Republic. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3705328.3759314
```
