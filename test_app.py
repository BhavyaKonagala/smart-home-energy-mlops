import unittest

from app import app


class TestSmartHomeAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_home(self):

        response = self.client.get("/")

        self.assertEqual(
            response.status_code,
            200
        )

    def test_health(self):

        response = self.client.get("/health")

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.get_json()

        self.assertEqual(
            data["status"],
            "healthy"
        )

    def test_prediction(self):

        sample = {
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
        }

        response = self.client.post(
            "/predict",
            json=sample
        )

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.get_json()

        self.assertIn(
            "prediction",
            data
        )

        self.assertIn(
            int(data["prediction"]),
            [0, 1]
        )


if __name__ == "__main__":
    unittest.main()
