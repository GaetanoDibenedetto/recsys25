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

