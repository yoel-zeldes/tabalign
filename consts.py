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

MODAL_VOLUME_NAME = "tabular-cache"

TABFM_DEFAULT_N_ESTIMATORS = 8
TABPFN_DEFAULT_N_ESTIMATORS = 8

DEFAULT_WEIGHT_DECAY = 0

DEFAULT_N_SAMPLES = 1000

