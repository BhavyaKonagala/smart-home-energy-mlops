import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists("smart_home_energy_raw.xlsx")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("smart_home_energy_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_processed_dataset_created(self):
        self.assertTrue(
            os.path.exists("smart_home_energy_processed.csv")
        )

    def test_accuracy_is_valid(self):

        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):

        model = joblib.load(
            "smart_home_energy_model.pkl"
        )

        sample = pd.DataFrame([{
            "Household_Size": 4,
            "Monthly_Income": 50000,
            "Home_Area_sqft": 1500,
            "AC_Hours_Daily": 8,
            "Refrigerator_Hours": 24,
            "Washing_Machine_Uses": 5,
            "TV_Hours_Daily": 5,
            "Computer_Hours_Daily": 4,
            "Lighting_Hours_Daily": 6,
            "Smart_Appliances": 3
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )


if __name__ == "__main__":
    unittest.main()
