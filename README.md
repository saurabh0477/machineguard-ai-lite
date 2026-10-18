# ⚙️ MachineGuard AI Lite
> **Predictive Maintenance for Industry (Hackathon Track: FT-03)**  
> *An industrial web application that predicts machine failure risk, explains mechanical stress factors, simulates operational digital twins, and calculates maintenance ROI.*

---

## 🌟 Executive Summary

In modern manufacturing, **unplanned machine downtime** causes catastrophic losses: destroyed cutting tools, scrapped workpieces, delayed shipments, and massive repair bills. Traditional factories rely either on **reactive maintenance** (waiting for machines to break) or **rigid calendar schedules** (servicing machines too early, wasting useful component life).

**MachineGuard AI Lite** bridges this gap with an intuitive, machine-learning-powered early warning dashboard. Using industrial operating telemetry, it:
1. Predicts impending machine failures before they occur.
2. Explains the mechanical factors driving risk (thermal dissipation, spindle overload, tool degradation).
3. Simulates **"what-if" operational adjustments** via the **Machine Scenario Lab** digital twin.
4. Calculates the expected **dollar return on investment (ROI)** of proactive maintenance vs. reactive breakdown.

---

## 🛠️ Technology Stack

- **Application Framework**: [Streamlit](https://streamlit.io/) (Clean industrial UI with dark navy sidebar, cards, and interactive widgets)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (`RandomForestClassifier`, `ColumnTransformer`, `Pipeline`)
- **Data Handling**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Visualizations**: [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Serialization**: [Joblib](https://joblib.readthedocs.io/)
- **Dataset**: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 real machine telemetry records)

---

## 🔬 Dataset & Machine Learning Architecture

### 1. Data Integrity & Leakage Prevention
The UCI AI4I 2020 dataset provides 10,000 observations reflecting typical milling machine telemetry. To ensure genuine machine learning:
- **Identifiers Excluded**: `UDI` and `Product ID` are dropped to prevent overfitting on arbitrary sequence indices.
- **Leakage Columns Excluded**: Specific failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) represent post-failure diagnostic outcomes. Training with them would cause 100% data leakage. They are strictly excluded from model inputs.

### 2. Physical Feature Engineering
In addition to raw telemetry, domain-informed physical metrics are computed:
- **Thermal Dissipation Differential**: $\Delta T = \text{Process Temperature [K]} - \text{Air Temperature [K]}$  
  *Detects heat buildup and cooling system degradation.*
- **Rotational Mechanical Power**: $P = \text{Torque [Nm]} \times \left(\text{Speed [rpm]} \times \frac{2\pi}{60}\right)$  
  *Detects motor overload or anomalous mechanical resistance.*

### 3. Class Imbalance & Evaluation
In the dataset, only **339 out of 10,000 machines (3.39%)** experience failures. A naive model could guess "Normal" and achieve 96.6% accuracy while missing every single breakdown.

To address this:
- **Stratified Split**: 80% training (8,000 records), 20% held-out test set (2,000 records).
- **Class Balancing**: `class_weight='balanced'` inside the Random Forest classifier.
- **Genuine Test-Set Results** (Evaluated strictly on held-out test data):
  - **Accuracy**: `98.55%`
  - **Precision**: `76.00%`
  - **Recall**: `83.82%` *(Catches 57 out of 68 actual test failures)*
  - **F1-Score**: `79.72%`
  - **Confusion Matrix**: `[[1914, 18], [11, 57]]` (TN, FP / FN, TP)

---

## 🚀 Key Dashboard Features

### A. 📊 Fleet Overview
- Live KPIs: Monitored fleet records, verified historical failure rate (3.39%), average torque, and tool wear.
- Interactive Plotly scatter plot mapping Torque vs. Spindle Speed with failure clustering.
- Failure distribution breakdown across machine quality variants (Type L, M, and H).
- Searchable and filterable telemetry log.

### B. 🔍 Failure Prediction
- Select an existing machine record or input manual sensor readings.
- Physical bounds validation and dual-scale temperature displays (Kelvin & Celsius).
- Outputs predicted state, calibrated failure probability %, and risk level (`LOW` < 25%, `MEDIUM` 25–50%, `HIGH` > 50%).
- Actionable industrial recommendations (e.g. *Halt & Dispatch Maintenance*, *Increase Monitoring Frequency*).
- Mechanical contributing factors explanation and global feature importance chart.

### C. 🔬 Machine Scenario Lab (Distinctive Feature)
- An operational digital twin that empowers engineers to test "what-if" operational adjustments without risk.
- Choose a baseline machine preset (e.g. Heavy Load & Worn Cutter, Thermal Inefficiency).
- Adjust sliders (speed, torque, tool wear, temperature) and instantly see:
  - Side-by-side comparison of baseline vs modified readings.
  - Calculated deltas ($\Delta \text{Wear}$, $\Delta \text{Torque}$, $\Delta \text{Risk}$).
  - Comparative visual probability gauge.

### D. 💰 Maintenance Cost ROI Calculator
- Interactive financial decision simulator comparing:
  - **Reactive Strategy**: $\text{Expected Loss} = P(\text{Failure}) \times \text{Breakdown Cost}$
  - **Proactive Strategy**: $\text{Expected Loss} = \text{PM Cost} + P_{\text{residual}}(\text{Failure}) \times \text{Breakdown Cost}$
  - **Net Savings**: $\text{Loss}_{\text{Reactive}} - \text{Loss}_{\text{Proactive}}$
- Visual bar chart and ROI percentage calculation for plant managers.

### E. 📈 Model Performance
- Transparent test-set accuracy, precision, recall, and F1-score.
- Interactive confusion matrix heatmap.
- Educational primer on why **Recall** matters when missed failures cost 100x more than false alarms.

---

## ⚡ Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Internet connection (for automatic one-time UCI dataset download)

### Step 1: Clone / Navigate to Project
```bash
cd machineguard-ai-lite
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```
*The app will automatically download the UCI dataset and initialize the model on first launch!*

*(Optional) To retrain the model independently from the command line:*
```bash
python train_model.py
```

---

## ☁️ Deployment Instructions

### Deploy to Streamlit Community Cloud (Recommended)
1. Initialize a git repository and push the project to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of MachineGuard AI Lite"
   git remote add origin https://github.com/<your-username>/machineguard-ai-lite.git
   git push -u origin main
   ```
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** and select your repository:
   - **Repository**: `<your-username>/machineguard-ai-lite`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!** The application will automatically install dependencies from `requirements.txt` and launch.

---

## 🛡️ Demonstration Disclaimers
*MachineGuard AI Lite is an educational and prototype demonstration system built for the FT-03 Hackathon. Predictions are generated using machine learning models trained on synthetic benchmark data (UCI AI4I 2020) and are designed to assist human technicians, not to autonomously command safety-critical industrial equipment.*
