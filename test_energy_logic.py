import unittest

from energy_logic import predict_energy_usage


class TestEnergyLogic(unittest.TestCase):

    def test_high_energy_usage(self):
        result = predict_energy_usage(10, 4)
        self.assertEqual(result, "HIGH")

    def test_low_energy_hours(self):
        result = predict_energy_usage(5, 4)
        self.assertEqual(result, "LOW")

    def test_low_smart_appliances(self):
        result = predict_energy_usage(10, 2)
        self.assertEqual(result, "LOW")


if __name__ == "__main__":
    unittest.main()
