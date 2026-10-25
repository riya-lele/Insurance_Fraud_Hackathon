# Hackathon add-ons spec
Holdout: 20 percent stratified (seed 42) held out from model and calibrator; remaining 80 percent split 75/25 into fit and calibration. Bundle models/final_v2.joblib has the same keys as final.joblib plus holdout_index.
Ranking uses raw_score. Zones use calibrated prob with t = c/(L+c): approve below t, review from t to 2t, investigate at 2t or above. c = 1, default L = 10. All costs are hypothetical.
Batch input columns = bundle["feature_list"]; FraudFound_P is optional and used only for evaluation.
Protected attributes for the fairness audit: Sex, MaritalStatus, Age, AgeOfPolicyHolder.