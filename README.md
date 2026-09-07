# ArthoSense NER: AI-Assisted Early Knee Osteoarthritis Screening Tool

[![SIH Problem Statement](https://img.shields.io/badge/SIH-SIH26004-blue.svg)](https://www.sih.gov.in/)
[![Offline First](https://img.shields.io/badge/Architecture-100%25%20Offline%20Edge-green.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Streamlit%20%7C%20OpenCV%20%7C%20MediaPipe-orange.svg)]()
[![Languages](https://img.shields.io/badge/Languages-7%20Regional%20Languages-purple.svg)]()

**ArthoSense NER** is a low-cost, offline-first multimodal diagnostic and screening tool designed for the early detection of Knee Osteoarthritis (OA) across remote and mountainous regions in the **North East Region (NER)** of India.

---

## 🏔️ Core Problem & Regional Context (SIH26004)

- **Geographic & Diagnostic Isolation:** Mountainous terrain in the North East Region restricts physical access to centralized hospital imaging (X-Ray / MRI).
- **High Occupational Mechanical Loading:** High prevalence in rural communities (e.g., tea-garden workers in Assam) due to prolonged squatting, uphill walking, and weight-bearing tasks.
- **Connectivity Constraints:** Severe internet instability demands a **100% offline edge architecture**.
- **Language Barriers:** Village healthcare workers (ASHAs/ANMs) require localized interfaces in regional languages.

---

## 💡 Key Features & System Architecture

```
                                  ┌──────────────────────────┐
                                  │  Multilingual Interface  │
                                  │ (7 Regional Languages)   │
                                  └────────────┬─────────────┘
                                               │
┌───────────────────────────────┐              ▼              ┌───────────────────────────────┐
│     Computer Vision (CV)      │   ┌─────────────────────┐   │       Wearable Sensors        │
│ • Google MediaPipe Pose       ├──►│   Dual-Engine AI    │◄──┤ • MPU6050 6-DOF IMU           │
│ • Real-time Knee ROM Angle    │   │ • Assam Rule Engine │   │ • Piezo Acoustic Contact Mic  │
│ • Extension/Flexion Kinematics│   │ • Supervised ML     │   │ • Joint Crepitus & Stability  │
└───────────────────────────────┘   └──────────┬──────────┘   └───────────────────────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │  Local SQLite Persistence│
                                  │  & Digital Report (PDF)  │
                                  └──────────────────────────┘
```

1. **Multimodal Data Capture & Perception:**
   - **Vision Kinematics:** Uses a standard laptop webcam with **Google MediaPipe Pose** to calculate real-time Knee Flexion Angles (Range of Motion).
   - **Wearable Sensors:** Ingests live kinematics from an ultra-low-cost ($5) **MPU6050 IMU** and **Piezo contact microphone** to detect joint micro-vibrations and acoustic crepitus.
2. **Explainable Clinical Expert System & ML:**
   - **Evidence-Informed Rule Engine:** Calibrated with published odds ratios from the **Assam Jorhat District Study** (Age >50 OR 4.53, Female OR 1.71), Eastern India studies (Menopause OR 3.15), and BMI (OR 1.86–5.29).
   - **Supervised Machine Learning:** Random Forest / Logistic Regression ensemble predicting calibrated risk probabilities.
   - **Risk Classification:** **Low Risk**, **Moderate Risk**, and **High Risk** mapped to actionable clinical care pathways.
3. **100% Offline Edge Persistence & Field Sync:**
   - Local SQLite database (`arthosense.db`) with one-click **CSV/JSON export** for periodic batch syncing via USB.
4. **Multilingual Regional Localization:**
   - Built-in support for **English, Assamese (অসমীয়া), Khasi (Ka Ktien Khasi), Manipuri (মৈতৈলোন্), Hindi, Mizo, and Bodo**.
5. **Digital Clinical Report Cards:**
   - Instant generation of printable **PDF** and **HTML** medical cards for PHC/District Hospital handoffs.

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ArthoSense_NER.git
cd ArthoSense_NER
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Local Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
ArthoSense_NER/
├── app.py                     # Streamlit Main Field Dashboard
├── clinical_engine.py         # Evidence-Informed Rule Engine & Supervised ML Classifier
├── vision_kinematics.py       # MediaPipe Pose tracking & Angle Kinematics
├── sensor_stream.py           # MPU6050 IMU + Piezo Ingestion & Hardware Synthesizer
├── database.py                # Local SQLite schema, CRUD operations & CSV/JSON export
├── localization.py            # Multilingual Regional Dictionary (7 Languages)
├── report_generator.py        # Digital Medical Report Card Generator (PDF & HTML)
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Standard Python git ignore
└── README.md                  # Project documentation & presentation guide
```

---

## 📊 Scientific & Epidemiological Basis

The clinical scoring engine incorporates empirical findings from:
- *Prevalence of primary osteoarthritis of knee in tea garden community of Jorhat District, Assam* (Age > 50 OR = 4.53, Female OR = 1.71).
- *Indian Community Studies on OA & Joint Pain in Eastern India* (Menopause OR = 3.15, Obesity OR = 1.86–5.29, Sedentary activity OR = 1.38).

---

## 👥 Smart India Hackathon (SIH) 
- **Problem Statement:** SIH26004 — AI-Assisted Early Detection System for Knee Osteoarthritis in the North East Region (NER)
- **Deployment Mode:** 100% Offline Edge Architecture for Rural Primary Health Centres (PHCs) and Village Sub-Centres.
colab - ayush - niraj