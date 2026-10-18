"""Result contract and signature checks. Run: python -m unittest discover -s tests."""
import json
import unittest

from crai import attestation, crypto, guard


class ResultContractTests(unittest.TestCase):
    def setUp(self):
        self.payload = guard.build_payload("a" * 64, "b" * 64, 0.0067, now=100)

    def test_deck_payload(self):
        self.assertEqual(self.payload, {
            "schema": "crai.result.v1", "study_hash": "a" * 64,
            "model_hash": "b" * 64, "pneumonia_probability": 0.0067, "time": 100,
        })

    def test_rejects_invalid_fields(self):
        for field, value in [("patient_id", "123"), ("image_png", "pixels"),
                             ("time", True), ("time", 0),
                             ("pneumonia_probability", "0.0067"),
                             ("pneumonia_probability", True),
                             ("pneumonia_probability", -0.1),
                             ("pneumonia_probability", 1.1),
                             ("pneumonia_probability", float("nan")),
                             ("pneumonia_probability", float("inf"))]:
            with self.subTest(field=field, value=value), self.assertRaises(guard.SchemaError):
                guard.validate_payload({**self.payload, field: value})
        old = dict(self.payload)
        old["issued_at"] = old.pop("time")
        with self.assertRaises(guard.SchemaError):
            guard.validate_payload(old)

    def test_rounding_and_nonfinite_model_output(self):
        self.assertEqual(guard.build_payload("a" * 64, "b" * 64, 0.123456, 100)
                         ["pneumonia_probability"], 0.1235)
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.assertRaises(guard.SchemaError):
                guard.build_payload("a" * 64, "b" * 64, value, 100)

    def test_signed_round_trip_and_tampering(self):
        issuer, guest = crypto.SigningKey.generate(), crypto.SigningKey.generate()
        claims = {"guest_image_hash": "c" * 64, "model_hash": "b" * 64,
                  "x25519_pub": crypto.ExchangeKey().public_hex(),
                  "ed25519_pub": guest.public_hex(), "nonce": "d" * 64}
        token = attestation.issue_token(issuer, claims, 60, now=100)
        kind, envelope = guard.parse_guest_output(guard.sign_envelope(self.payload, guest, token))
        self.assertEqual(kind, "result")
        checks = guard.verify_envelope(envelope, issuer.public_hex(), "b" * 64, "a" * 64)
        self.assertTrue(all(checks.values()), checks)
        for field, value in [("time", 101), ("pneumonia_probability", 0.5)]:
            altered = json.loads(json.dumps(envelope))
            altered["payload"][field] = value
            self.assertFalse(guard.verify_envelope(altered, issuer.public_hex(), "b" * 64,
                                                  "a" * 64)["payload_signature"])


if __name__ == "__main__":
    unittest.main()
