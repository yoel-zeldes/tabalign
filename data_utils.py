import numpy as np
import pandas as pd
import openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tabpfn.preprocessing.modality_detection import detect_feature_modalities
from tabpfn.preprocessing.clean import clean_data, fix_dtypes, process_text_na_dataframe
from tabpfn.preprocessing.datamodel import FeatureModality

# TabArena-v0.1 benchmark classification datasets - without regression datasets (OpenML suite 457).
# Maps dataset name -> OpenML task_id.
# task_ids are from: https://www.openml.org/api/v1/json/study/457
# dataset names as well as classification/regression categorization are from: https://github.com/TabArena/tabarena_dataset_curation/blob/main/dataset_creation_scripts/metadata/created_datasets.json
TABARENA_NAME_TO_TASK_ID = {
    "Amazon_employee_access": 363613,
    "anneal": 363614,
    # "APSFailure": 363616,
    "bank-marketing": 363618,
    "Bank_Customer_Churn": 363619,
    # "Bioresponse": 363620,
    "blood-transfusion-service-center": 363621,
    "churn": 363623,
    "coil2000_insurance_policies": 363624,
    "credit-g": 363626,
    "credit_card_clients_default": 363627,
    "customer_satisfaction_in_airline": 363628,
    "diabetes": 363629,
    "Diabetes130US": 363630,
    "E-CommereShippingData": 363632,
    "Fitness_Club": 363671,
    "GiveMeSomeCredit": 363673,
    "hazelnut-spread-contaminant-detection": 363674,
    "heloc": 363676,
    # "hiva_agnostic": 363677,
    "HR_Analytics_Job_Change_of_Data_Scientists": 363679,
    "in_vehicle_coupon_recommendation": 363681,
    "Is-this-a-good-customer": 363682,
    "jm1": 363712,
    # "kddcup09_appetency": 363683,
    "Marketing_Campaign": 363684,
    "maternal_health_risk": 363685,
    # "MIC": 363711,
    "NATICUSdroid": 363689,
    "online_shoppers_intention": 363691,
    "polish_companies_bankruptcy": 363694,
    "qsar-biodeg": 363696,
    "SDSS17": 363699,
    "seismic-bumps": 363700,
    "splice": 363702,
    "students_dropout_and_academic_success": 363704,
    "taiwanese_bankruptcy_prediction": 363706,
    "website_phishing": 363707,
}


def stratified_subsample(X, y, size, seed):
    """Stratified subsampling to preserve class balance when truncating.
    Without this, datasets with sorted indices (e.g. TabArena) lose minority classes."""
    X_sub, X_rest, y_sub, y_rest = train_test_split(
        X, y, train_size=size, stratify=y, random_state=seed
    )
    return X_sub, y_sub, X_rest, y_rest


def load_raw_data(dataset_name, repeat=0, max_num_examples=1000):
    if max_num_examples > 10000:
        raise ValueError("max_num_examples must be less than or equal to 10000, because that's how TabPFN was trained")
    if not dataset_name.startswith("tabarena/"):
        raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    task = openml.tasks.get_task(TABARENA_NAME_TO_TASK_ID[dataset_name.removeprefix("tabarena/")])
    X, y = task.get_X_and_y(dataset_format="dataframe")
    y = LabelEncoder().fit_transform(y.astype(str))
    train_idx, test_idx = task.get_train_test_split_indices(fold=0, repeat=repeat)
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    if len(X_test) > 500:
        X_test, y_test, *_ = stratified_subsample(X_test, y_test, 500, seed=2)
    
    if len(X_train) > max_num_examples:
        X_train, y_train, *_ = stratified_subsample(X_train, y_train, max_num_examples, seed=3)
    n_unique_labels = len(np.unique(y_train))
    if n_unique_labels >= 30:
        raise ValueError(
            f"Dataset '{dataset_name}' has {n_unique_labels} unique labels, so it's probably not a classification dataset."
        )

    # Preprocess categorical/text features into numeric float arrays fit on train
    # and applied to test, ensuring compatibility across all models (TabPFN, TabFM, XGBoost).
    feature_schema = detect_feature_modalities(
        X=X_train.values,
        feature_names=list(X_train.columns),
        min_samples_for_inference=100,
        max_unique_for_category=30,
        min_unique_for_numerical=4,
        min_cardinality_for_text=30,
    )
    X_train_clean, ord_encoder, feature_schema = clean_data(X_train.values, feature_schema)
    inferred_cat_indices = feature_schema.indices_for(FeatureModality.CATEGORICAL)
    X_test_clean = fix_dtypes(pd.DataFrame(X_test.values), cat_indices=inferred_cat_indices)
    X_test_clean = process_text_na_dataframe(X_test_clean, ord_encoder=ord_encoder)
    return X_train_clean, X_test_clean, y_train, y_test, inferred_cat_indices
