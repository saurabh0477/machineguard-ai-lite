import pandas as pd
import joblib
from app import (
    load_default_dataset,
    load_or_train_model,
    predict_single_record,
    categorize_risk,
    generate_plain_english_explanation,
    suggest_next_step,
)

print("1. Testing dataset load...")
df = load_default_dataset()
assert df is not None and len(df) == 10000, f"Expected 10000 rows, got {len(df) if df is not None else None}"
print(f"   Dataset loaded: {len(df)} rows.")

print("2. Testing model artifact load...")
artifact = load_or_train_model(None)
assert artifact is not None and "pipeline" in artifact, "Artifact or pipeline missing"
pipeline = artifact["pipeline"]
print("   Model loaded successfully.")

print("3. Testing prediction on healthy input...")
healthy_input = {
    "Type": "M",
    "Air temperature [K]": 298.0,
    "Process temperature [K]": 308.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 20,
}
p_class, p_prob, eng = predict_single_record(pipeline, healthy_input)
risk, badge, sub = categorize_risk(p_prob)
rec = suggest_next_step(risk)
print(f"   Healthy input -> Class: {p_class}, Prob: {p_prob:.4f}, Risk: {risk}, Action: {rec['action']}")

print("4. Testing prediction on failure-prone input...")
faulty_input = {
    "Type": "L",
    "Air temperature [K]": 303.0,
    "Process temperature [K]": 311.0,
    "Rotational speed [rpm]": 1300,
    "Torque [Nm]": 65.0,
    "Tool wear [min]": 220,
}
f_class, f_prob, f_eng = predict_single_record(pipeline, faulty_input)
f_risk, f_badge, f_sub = categorize_risk(f_prob)
f_rec = suggest_next_step(f_risk)
reasons = generate_plain_english_explanation(f_eng, f_prob)
print(f"   Faulty input -> Class: {f_class}, Prob: {f_prob:.4f}, Risk: {f_risk}, Action: {f_rec['action']}")
print("   Reasons:")
for r in reasons:
    print(f"     - {r}")

print("All validation checks passed successfully!")
