"""
ArthoSense NER - Clinical Expert System & Machine Learning Engine
Grounded in empirical epidemiological evidence from the Assam (Jorhat District) study
and Indian population-level Osteoarthritis research.

Combines:
1. Evidence-Informed Rule-Based Scoring Engine (illustrative clinical weighted scoring)
2. Supervised Machine Learning Classifier (Random Forest / Logistic Regression)
   for calibrated risk probability estimation and feature importance attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix

# ==============================================================================
# 1. EPIDEMIOLOGICAL EVIDENCE WEIGHTS & ODDS RATIO REFERENCE (ASSAM & INDIAN STUDIES)
# ==============================================================================
# - Age > 50: Assam Jorhat Tea-Garden Study OR = 4.53 (95% CI 2.54-8.06)
# - Female Sex: Assam Jorhat Study OR = 1.71 (95% CI 1.02-2.86)
# - Menopause: Eastern India Women Study OR = 3.15 (75.4% aged 50-59, 80% menopausal)
# - High BMI / Obesity: Indian Community Studies OR = 1.86 - 5.29
# - Heavy Occupational Loading: Assam Tea-Garden mechanical knee loading (squatting, sloping terrain)
# - Previous Knee Trauma/Injury: Pooled OR = 1.51 - 3.86
# - Family History of OA: OR = 1.61 (familial HR ~1.75)
# - Sedentary Lifestyle: Indian Rural Studies OR = 1.38 - 3.26 (36.8% vs 26.6%)
# ==============================================================================

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculates Body Mass Index (BMI = kg / m^2)."""
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 1)

def compute_evidence_rule_score(
    age: int,
    gender: str,
    bmi: float,
    menopause: bool,
    previous_injury: bool,
    family_history: bool,
    activity_level: str,
    occupation_loading: bool,
    pain_score: int,
    stiffness_symptom: bool,
    crepitus_symptom: bool,
    rom_angle: float,
    vibration_rms: float
) -> Dict[str, Any]:
    """
    Computes evidence-informed risk score (0-100) using weights derived from
    Assam & Indian epidemiological research, kinematic ROM, and acoustic vibration metrics.
    """
    points = 0.0
    max_possible_points = 100.0
    breakdown = {}

    # 1. Age Factor (Assam OR = 4.53)
    if age > 60:
        age_pts = 16.0
        age_note = f"Age {age} (>60 yr): High epidemiological risk (Assam OR 4.53)"
    elif age > 50:
        age_pts = 12.0
        age_note = f"Age {age} (>50 yr): Significant risk (Assam OR 4.53)"
    elif age >= 40:
        age_pts = 6.0
        age_note = f"Age {age} (40-50 yr): Moderate baseline risk"
    else:
        age_pts = 2.0
        age_note = f"Age {age} (<40 yr): Low baseline risk"
    points += age_pts
    breakdown["Age Factor"] = {"points": age_pts, "max": 16.0, "note": age_note}

    # 2. Gender & Menopause (Female OR = 1.71, Menopause OR = 3.15)
    gender_pts = 0.0
    if gender.lower() == "female":
        gender_pts += 4.0
        if menopause:
            gender_pts += 6.0
            g_note = "Female sex (OR 1.71) + Postmenopausal status (OR 3.15)"
        else:
            g_note = "Female sex (Assam OR 1.71)"
    else:
        g_note = "Male / Reference baseline"
    points += gender_pts
    breakdown["Sex & Hormonal"] = {"points": gender_pts, "max": 10.0, "note": g_note}

    # 3. BMI & Obesity (Indian cutoffs: >=25 Overweight, >=30 Obese, OR 1.86-5.29)
    if bmi >= 30.0:
        bmi_pts = 12.0
        bmi_note = f"BMI {bmi} (Obese class): Severe joint load factor (OR 1.86-5.29)"
    elif bmi >= 25.0:
        bmi_pts = 7.0
        bmi_note = f"BMI {bmi} (Overweight): Increased mechanical cartilage stress"
    elif bmi >= 18.5:
        bmi_pts = 2.0
        bmi_note = f"BMI {bmi} (Normal range): Healthy weight distribution"
    else:
        bmi_pts = 0.0
        bmi_note = f"BMI {bmi} (Underweight)"
    points += bmi_pts
    breakdown["BMI & Weight Load"] = {"points": bmi_pts, "max": 12.0, "note": bmi_note}

    # 4. Heavy Occupational Loading (Assam Tea Garden manual work: standing, squatting, slopes)
    occ_pts = 0.0
    if occupation_loading:
        occ_pts = 10.0
        occ_note = "Tea-garden / Heavy manual occupational mechanical loading identified"
    else:
        occ_note = "Standard occupational loading"
    points += occ_pts
    breakdown["Occupational Knee Loading"] = {"points": occ_pts, "max": 10.0, "note": occ_note}

    # 5. History of Trauma/Injury & Genetics (Injury OR 1.51-3.86, Family OR 1.61)
    hx_pts = 0.0
    hx_notes = []
    if previous_injury:
        hx_pts += 6.0
        hx_notes.append("Previous knee trauma (OR 1.51-3.86)")
    if family_history:
        hx_pts += 4.0
        hx_notes.append("Positive family history (OR 1.61)")
    if not hx_notes:
        hx_notes.append("No previous trauma or family history")
    points += hx_pts
    breakdown["Injury & Family History"] = {"points": hx_pts, "max": 10.0, "note": "; ".join(hx_notes)}

    # 6. Physical Inactivity / Sedentary (Prevalence 36.8% vs 26.6%)
    act_pts = 0.0
    if activity_level.lower() == "sedentary":
        act_pts = 5.0
        act_note = "Sedentary activity pattern (Higher OA prevalence 36.8%)"
    elif activity_level.lower() == "moderate":
        act_pts = 2.0
        act_note = "Moderate daily physical activity"
    else:
        act_note = "Active lifestyle (Protective joint movement)"
    points += act_pts
    breakdown["Physical Activity"] = {"points": act_pts, "max": 5.0, "note": act_note}

    # 7. Clinical Symptoms (Pain VAS 0-10, Morning Stiffness >30 min, Crepitus history)
    sx_pts = 0.0
    # Pain score (0-10) scaled to 7 pts
    sx_pts += min(7.0, (pain_score / 10.0) * 7.0)
    if stiffness_symptom:
        sx_pts += 4.0
    if crepitus_symptom:
        sx_pts += 4.0
    points += sx_pts
    breakdown["Symptom Profile"] = {
        "points": round(sx_pts, 1),
        "max": 15.0,
        "note": f"Pain VAS {pain_score}/10, Stiffness: {'Yes' if stiffness_symptom else 'No'}, Crepitus history: {'Yes' if crepitus_symptom else 'No'}"
    }

    # 8. Computer Vision Kinematic ROM (Webcam Angle)
    # Normal >130 deg, Mild 110-130 deg, Moderate 90-110 deg, Severe <90 deg
    if rom_angle < 90.0:
        rom_pts = 12.0
        rom_note = f"ROM {rom_angle:.1f}°: Severe flexion restriction (<90°)"
    elif rom_angle < 110.0:
        rom_pts = 8.0
        rom_note = f"ROM {rom_angle:.1f}°: Moderate flexion restriction (90°-110°)"
    elif rom_angle < 130.0:
        rom_pts = 4.0
        rom_note = f"ROM {rom_angle:.1f}°: Mild joint tightness (110°-130°)"
    else:
        rom_pts = 0.0
        rom_note = f"ROM {rom_angle:.1f}°: Normal full range of motion (>=130°)"
    points += rom_pts
    breakdown["Vision Kinematic ROM"] = {"points": rom_pts, "max": 12.0, "note": rom_note}

    # 9. Piezo Acoustic Crepitus Vibration (RMS 0.0 to 1.0+)
    if vibration_rms >= 0.70:
        vib_pts = 10.0
        vib_note = f"Vibration RMS {vibration_rms:.2f}: Severe acoustic crepitus grinding"
    elif vibration_rms >= 0.35:
        vib_pts = 6.0
        vib_note = f"Vibration RMS {vibration_rms:.2f}: Moderate joint micro-vibration"
    elif vibration_rms >= 0.15:
        vib_pts = 2.0
        vib_note = f"Vibration RMS {vibration_rms:.2f}: Slight physiological clicking"
    else:
        vib_pts = 0.0
        vib_note = f"Vibration RMS {vibration_rms:.2f}: Clean acoustic profile (Smooth)"
    points += vib_pts
    breakdown["Acoustic Crepitus & Vibration"] = {"points": vib_pts, "max": 10.0, "note": vib_note}

    # Normalization
    risk_score = round(min(100.0, max(0.0, points)), 1)

    # Classification into Low / Moderate / High Risk and Clinical Referral Pathways (Aligning with SIH PDF workflow)
    if risk_score >= 60.0:
        risk_category = "High Risk"
        oa_grade = "High Risk"
        referral_tier = "Clinical Evaluation Recommended (Referral to District Hospital / Orthopedic Specialist)"
    elif risk_score >= 35.0:
        risk_category = "Moderate Risk"
        oa_grade = "Moderate Risk"
        referral_tier = "Assessment & Physiotherapy (Primary Health Centre - PHC / Lifestyle Modifications)"
    else:
        risk_category = "Low Risk"
        oa_grade = "Low Risk"
        referral_tier = "Monitor (Routine Wellness & Annual Sub-Centre Screening)"

    return {
        "risk_score": risk_score,
        "risk_category": risk_category,
        "oa_grade": oa_grade,
        "referral_tier": referral_tier,
        "breakdown": breakdown,
        "raw_points": points
    }


# ==============================================================================
# 2. SUPERVISED MACHINE LEARNING CLASSIFIER MODULE
# ==============================================================================

class OAClassifierEngine:
    """
    Supervised Machine Learning module for Knee Osteoarthritis risk prediction.
    Trained on clinical risk vectors [Age, BMI, Female, Menopause, Injury, 
    Family_History, Sedentary, Heavy_Loading, Pain_Score, ROM_Angle, Vibration_RMS].
    """

    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
        self.feature_names = [
            "Age", "BMI", "Female_Sex", "Menopause", "Previous_Injury",
            "Family_History", "Sedentary_Activity", "Heavy_Knee_Loading",
            "Pain_Score", "ROM_Angle", "Vibration_RMS"
        ]
        self.is_trained = False
        self.metrics = {}
        self._train_baseline_cohort()

    def _generate_synthetic_cohort(self, n_samples: int = 1200) -> pd.DataFrame:
        """
        Synthesizes a representative epidemiological cohort grounded in the
        Assam Jorhat study and Eastern India distributions for model pre-training.
        """
        np.random.seed(42)
        ages = np.random.normal(52, 14, n_samples).clip(20, 85)
        bmis = np.random.normal(25.5, 4.8, n_samples).clip(16.0, 42.0)
        females = np.random.binomial(1, 0.55, n_samples)
        
        # Menopause depends on age and female sex
        menopause = np.where((females == 1) & (ages >= 48), np.random.binomial(1, 0.82, n_samples), 0)
        previous_injury = np.random.binomial(1, 0.22, n_samples)
        family_history = np.random.binomial(1, 0.28, n_samples)
        sedentary = np.random.binomial(1, 0.35, n_samples)
        heavy_loading = np.random.binomial(1, 0.45, n_samples) # Tea garden worker proportion
        
        # Latent log-odds based on published Odds Ratios
        # log(OR) weights:
        log_odds = (
            -5.2
            + 0.08 * (ages - 45)                           # Age > 50 OR ~ 4.53
            + 0.12 * (bmis - 23.0)                         # BMI OR ~ 1.86-5.29
            + 0.54 * females                               # Female OR ~ 1.71
            + 1.15 * menopause                             # Menopause OR ~ 3.15
            + 0.70 * previous_injury                       # Injury OR ~ 2.0
            + 0.48 * family_history                        # Family history OR ~ 1.61
            + 0.40 * sedentary                             # Sedentary OR ~ 1.38
            + 0.65 * heavy_loading                         # Heavy mechanical load
        )
        
        # Add random noise
        probs = 1.0 / (1.0 + np.exp(-log_odds))
        oa_labels = np.random.binomial(1, probs)

        # Derived physical metrics correlated with OA label
        pain_scores = np.where(oa_labels == 1, np.random.normal(6.5, 1.8, n_samples), np.random.normal(1.8, 1.5, n_samples)).clip(0, 10).round().astype(int)
        rom_angles = np.where(oa_labels == 1, np.random.normal(96, 15, n_samples), np.random.normal(132, 10, n_samples)).clip(65, 150).round(1)
        vibration_rms = np.where(oa_labels == 1, np.random.normal(0.72, 0.22, n_samples), np.random.normal(0.18, 0.12, n_samples)).clip(0.02, 1.45).round(2)

        df = pd.DataFrame({
            "Age": ages.round(1),
            "BMI": bmis.round(1),
            "Female_Sex": females,
            "Menopause": menopause,
            "Previous_Injury": previous_injury,
            "Family_History": family_history,
            "Sedentary_Activity": sedentary,
            "Heavy_Knee_Loading": heavy_loading,
            "Pain_Score": pain_scores,
            "ROM_Angle": rom_angles,
            "Vibration_RMS": vibration_rms,
            "OA_Label": oa_labels
        })
        return df

    def _train_baseline_cohort(self):
        """Trains models on the baseline calibrated dataset."""
        df = self._generate_synthetic_cohort()
        X = df[self.feature_names]
        y = df["OA_Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        self.rf_model.fit(X_train, y_train)
        self.lr_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)

        # Evaluate Random Forest
        y_pred = self.rf_model.predict(X_test)
        y_prob = self.rf_model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 3),
            "sensitivity": round(recall_score(y_test, y_pred) * 100, 1),
            "precision": round(precision_score(y_test, y_pred) * 100, 1),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test)
        }
        self.is_trained = True

    def predict_risk(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts patient features and returns estimated probability of OA and feature contributions.
        """
        vector_df = pd.DataFrame([[
            float(feature_dict.get("age", 45)),
            float(feature_dict.get("bmi", 24.0)),
            1.0 if str(feature_dict.get("gender", "")).lower() == "female" else 0.0,
            1.0 if feature_dict.get("menopause", False) else 0.0,
            1.0 if feature_dict.get("previous_injury", False) else 0.0,
            1.0 if feature_dict.get("family_history", False) else 0.0,
            1.0 if str(feature_dict.get("activity_level", "")).lower() == "sedentary" else 0.0,
            1.0 if feature_dict.get("occupation_loading", False) else 0.0,
            float(feature_dict.get("pain_score", 0)),
            float(feature_dict.get("rom_angle", 130.0)),
            float(feature_dict.get("vibration_rms", 0.1))
        ]], columns=self.feature_names)

        rf_prob = float(self.rf_model.predict_proba(vector_df)[0, 1])
        lr_prob = float(self.lr_model.predict_proba(vector_df)[0, 1])
        gb_prob = float(self.gb_model.predict_proba(vector_df)[0, 1])

        # Feature importances from Random Forest
        importances = dict(zip(self.feature_names, [round(float(x), 3) for x in self.rf_model.feature_importances_]))

        return {
            "ml_probability": round(rf_prob * 100, 1),
            "lr_probability": round(lr_prob * 100, 1),
            "gb_probability": round(gb_prob * 100, 1),
            "feature_importances": importances,
            "metrics": self.metrics
        }

# Global singleton classifier instance
classifier = OAClassifierEngine()
