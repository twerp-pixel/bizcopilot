import unittest

from agent.orchestrator import _validate_plan, _verify


class TestPlanValidation(unittest.TestCase):
    def test_accepts_valid_plan(self):
        plan = {
            "steps": [
                {"tool": "sql", "args": {"query": "SELECT 1"}},
                {"tool": "retrieve", "args": {"query": "sales notes"}},
            ]
        }
        self.assertEqual(_validate_plan(plan), plan)

    def test_rejects_string_step(self):
        plan = {
            "steps": [
                {"tool": "sql", "args": {"query": "SELECT 1"}},
                "tool",
            ]
        }
        with self.assertRaises(ValueError):
            _validate_plan(plan)

    def test_rejects_missing_tool(self):
        plan = {
            "steps": [
                {"tool": "sql", "args": {"query": "SELECT 1"}},
                {"tool": "sql", "args": {"query": "SELECT 2"}},
            ]
        }
        with self.assertRaises(ValueError):
            _validate_plan(plan)


class TestNumericVerification(unittest.TestCase):
    def test_accepts_numbers_present_in_tool_results(self):
        tool_results = [
            {
                "tool": "sql",
                "result": [
                    {"units": 673, "revenue": 10283.44},
                    {
                        "units": 465,
                        "revenue": 7105.20,
                        "revenue_pct_change": -30.91,
                    },
                ],
            }
        ]
        memo = "Units fell from 673 to 465. Revenue was 7105.20, down -30.91%."
        grounded, unverified = _verify(memo, tool_results)

        self.assertTrue(grounded)
        self.assertEqual(unverified, [])

    def test_ignores_numbered_list_prefixes(self):
        from agent.orchestrator import _extract_numbers

        text = (
            "1. Monitor the trend.\n"
            "2. Review sales for week 9."
        )
        numbers = _extract_numbers(text)

        self.assertNotIn("1", numbers)
        self.assertNotIn("2", numbers)
        self.assertIn("9", numbers)


    def test_rejects_unsigned_value_when_only_negative_change_is_supported(self):
        tool_results = [
            {"tool": "sql", "result": [{"pct_change": -30.91}]}
        ]
        grounded, unverified = _verify(
            "Revenue changed by 30.91%.", tool_results
        )
        self.assertFalse(grounded)
        self.assertIn("30.91", unverified)

    def test_accepts_two_decimal_rounding_of_source_percentage(self):
        tool_results = [
            {"tool": "sql", "result": [{"pct_change": -67.5373134328}]}
        ]
        grounded, unverified = _verify(
            "Units changed by -67.54%.", tool_results
        )
        self.assertTrue(grounded)
        self.assertEqual(unverified, [])



if __name__ == "__main__":
    unittest.main()
