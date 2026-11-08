import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure service directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "service"))

import app as service_app


class TestService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["S3_ACCESS_KEY"] = "admin"
        os.environ["S3_SECRET_KEY"] = "password"
        os.environ["MODEL_VERSION"] = "v1"
        cls.client = TestClient(service_app.app)

    def test_01_healthz_endpoint(self):
        """Test 1: GET /healthz returns 200 while process is alive."""
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "alive"})

    def test_02_readyz_endpoint_states(self):
        """Test 2: GET /readyz returns 503 before loading and 200 when ready."""
        service_app.READY = False
        res_not_ready = self.client.get("/readyz")
        self.assertEqual(res_not_ready.status_code, 503)

        service_app.READY = True
        res_ready = self.client.get("/readyz")
        self.assertEqual(res_ready.status_code, 200)

    def test_03_fault_inject_returns_500(self):
        """Test 3: When FAULT_INJECT=1, /parse returns HTTP 500 with exact error body."""
        service_app.READY = True
        with patch.dict(os.environ, {"FAULT_INJECT": "1"}):
            response = self.client.post("/parse", json={"text": "set an alarm for seven am"})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {"error": "invalid_model_output"})
            self.assertIn("X-Model-Version", response.headers)
            self.assertIn("X-Served-By", response.headers)

    def test_04_parse_valid_output_and_headers(self):
        """Test 4: POST /parse with valid model prediction returns HTTP 200 and exact 4 keys."""
        service_app.READY = True
        valid_prediction = '{"intent": "alarm_set", "date": "tomorrow", "time": "seven am", "place": null}'

        with patch.dict(os.environ, {"FAULT_INJECT": "0"}), patch.object(
            service_app, "predict", return_value=valid_prediction
        ):
            response = self.client.post("/parse", json={"text": "set an alarm for seven am tomorrow"})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(set(data.keys()), {"intent", "date", "time", "place"})
            self.assertEqual(data["intent"], "alarm_set")
            self.assertEqual(data["time"], "seven am")
            self.assertEqual(response.headers["X-Model-Version"], service_app.MODEL_VERSION)
            self.assertEqual(response.headers["X-Served-By"], service_app.POD_NAME)

    def test_05_schema_validation_rejects_invalid_schema(self):
        """Test 5: POST /parse rejects output with invalid schema (e.g. invalid intent not in schema)."""
        service_app.READY = True
        invalid_intent_prediction = '{"intent": "unrecognized_intent", "date": null, "time": "six am", "place": null}'

        with patch.dict(os.environ, {"FAULT_INJECT": "0"}), patch.object(
            service_app, "predict", return_value=invalid_intent_prediction
        ):
            response = self.client.post("/parse", json={"text": "wake me up at six am"})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {"error": "invalid_model_output"})


if __name__ == "__main__":
    unittest.main()
