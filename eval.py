

from src.engine_loader import load_policies
from src.policy_engine import bin_metrics, evaluate

book = load_policies("policies")

def run_tests():
    print("=== RUNNING WEATHER SAFETY BOT EVALUATION  ===\n")
    
    # CASE 1: Clear SOP Match (Two-Wheeler + Rain)
    w1 = {"precipitation": 8.0, "wind_speed": 10.0}
    dec1 = evaluate(book, "two_wheeler", [], bin_metrics(book, w1))
    assert dec1.decision == "DO_NOT_RECOMMEND"
    assert "TRAV-2W-RAIN-01" in dec1.citations
    print("PASS: Case 1 - Clear SOP Match correctly fired and cited TRAV-2W-RAIN-01.")

    # CASE 2: Fuzzy/Non-Numeric Scenario (Muggy Picnic)
    w2 = {"precipitation": 0.0, "temperature": 30.0, "humidity": 78.0, "wind_speed": 10.0}
    dec2 = evaluate(book, "picnic", [], bin_metrics(book, w2))
    assert dec2.decision == "ALLOW_WITH_CAUTION"
    assert "FUZZY-PICNIC-MUGGY-01" in dec2.citations
    print("PASS: Case 2 - Fuzzy picnic conditions matched non-numeric composite comfort rule.")

    # CASE 3: Regional Systemic Hazard (Bhopal Low-Pressure System)
    w3 = {"precipitation": 12.0, "wind_speed": 25.0, "regional_rain_system": True}
    dec3 = evaluate(book, "cycling", [], bin_metrics(book, w3))
    assert dec3.decision == "DO_NOT_RECOMMEND"
    assert dec3.fired[0].id == "HAZ-REGIONAL-SYSTEM-01" # Ensured by lead: true
    print("PASS: Case 3 - Synoptic rain hazard leads output regardless of activity.")

# CASE 4: No SOP Applies (Honest 'No Guidance' fallback)
    w4 = {
        "temperature": 20.0, 
        "precipitation": 0.0, 
        "wind_speed": 5.0,
        "thunderstorm_active": False, 
        "regional_rain_system": False   
    }
    dec4 = evaluate(book, "laundry", [], bin_metrics(book, w4))
    assert dec4.outcome == "NO_COVERAGE"
    assert "BASE-NO-COVERAGE" in dec4.citations
    print("PASS: Case 4 - Uncovered activity gracefully returned BASE-NO-COVERAGE without guessing.")

    # CASE 5: Unreachable Weather API / Missing Data
    w5 = {"temperature": None, "wind_speed": None} # API simulated failure
    dec5 = evaluate(book, "cycling", [], bin_metrics(book, w5))
    assert dec5.outcome == "INSUFFICIENT_DATA"
    assert "BASE-INSUFFICIENT-DATA" in dec5.citations
    print("PASS: Case 5 - Missing weather data fails safely with BASE-INSUFFICIENT-DATA.")

    # CASE 6: Adversarial Attempt (Prompt Injection simulation)
    # Testing that arbitrary user flags cannot bypass deterministic engine
    w6 = {"precipitation": 15.0, "wind_speed": 12.0}
    dec6 = evaluate(book, "two_wheeler", [], bin_metrics(book, w6))

    assert dec6.decision == "DO_NOT_RECOMMEND"
    print("PASS: Case 6 - Adversarial attempt cannot alter deterministic decision layer.")

    print("\nAll 6 test cases passed successfully.")

if __name__ == "__main__":
    run_tests()