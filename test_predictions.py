import requests

# Test cases with more realistic profiles
test_cases = [
    {
        # Very High Risk Customer
        "customer_id": "CUST007",
        "tenure": 2,                     # Very new customer
        "monthly_charges": 150.0,        # Very high charges
        "total_charges": 300.0,
        "contract_type": "Basic",        # Basic plan
        "tech_support": "No",            # No support
        "internet_service": "Fiber optic", # High-end service
        "churn": 0,
        "Age": 25,                       # Young customer
        "Gender": "Male",
        "Payment_Delay": 15              # High payment delay
    },
    {
        # Very Low Risk Customer
        "customer_id": "CUST008",
        "tenure": 72,                    # 6 years customer
        "monthly_charges": 65.0,         # Low charges
        "total_charges": 4680.0,
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
        response.raise_for_status()  # Raise an error for bad status codes
        
        result = response.json()
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
        
        if 'churn_probability' in result:
            print(f"Churn Probability: {result['churn_probability']:.2%}")
        else:
            print(f"Error in response: {result}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
    except KeyError as e:
        print(f"Error in response format: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}") 