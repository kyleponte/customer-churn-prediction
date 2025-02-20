import requests

# Test cases
test_cases = [
    {
        # High Risk Customer (for comparison)
        "customer_id": "CUST009",
        "tenure": 1,                     # Brand new customer
        "monthly_charges": 200.0,        # Very high charges
        "total_charges": 200.0,
        "contract_type": "Basic",        # Basic plan
        "tech_support": "No",            # No support
        "internet_service": "Fiber optic", # High-end service
        "churn": 0,
        "Age": 22,                       # Very young customer
        "Gender": "Male",
        "Payment_Delay": 30              # Severe payment delay
    },
    {
        # Medium Risk Customer
        "customer_id": "CUST011",
        "tenure": 12,                    # 1 year customer
        "monthly_charges": 120.0,        # Medium charges
        "total_charges": 1440.0,
        "contract_type": "Standard",     # Standard plan
        "tech_support": "Yes",           # Has support
        "internet_service": "Fiber optic", # High-end service
        "churn": 0,
        "Age": 35,                       # Middle-aged customer
        "Gender": "Male",
        "Payment_Delay": 5               # Minor payment delay
    },
    {
        # Low Risk Customer
        "customer_id": "CUST010",
        "tenure": 60,                    # Long-term customer (5 years)
        "monthly_charges": 70.0,         # Low charges
        "total_charges": 4200.0,
        "contract_type": "Premium",      # Premium plan
        "tech_support": "Yes",           # Has support
        "internet_service": "DSL",       # Basic service
        "churn": 0,
        "Age": 45,                       # Middle-aged customer
        "Gender": "Female",
        "Payment_Delay": 0               # No payment delays
    }
]

# Test each case
for case in test_cases:
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=case
        )
        response.raise_for_status()
        
        print(f"\nCustomer {case['customer_id']}:")
        print(f"Profile:")
        print(f"  - Age: {case['Age']} years")
        print(f"  - Gender: {case['Gender']}")
        print(f"  - Contract: {case['contract_type']}")
        print(f"  - Tenure: {case['tenure']} months")
        print(f"  - Monthly charges: ${case['monthly_charges']}")
        print(f"  - Payment Delay: {case['Payment_Delay']} days")
        print(f"  - Tech support: {case['tech_support']}")
        print(f"  - Internet service: {case['internet_service']}")
        print(f"Churn Probability: {response.json()['churn_probability']:.2%}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error details: {e.response.text}") 