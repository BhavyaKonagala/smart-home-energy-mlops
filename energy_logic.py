def predict_energy_usage(energy_hours, smart_appliances):
    if energy_hours >= 8 and smart_appliances >= 3:
        return "HIGH"
    else:
        return "LOW"


if __name__ == "__main__":
    result = predict_energy_usage(10, 4)
    print("Predicted Energy Usage:", result)
