import unittest

from faircheck.search import FairCheckSearchService, tokenize


class SearchServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = FairCheckSearchService()

    def test_tokenize_korean_query(self):
        self.assertEqual(tokenize("건설기계 임대단가 담합"), ["건설기계", "임대단가", "담합"])

    def test_search_decisions_returns_document_level_results(self):
        results = self.service.search_decisions("건설기계 임대단가 담합", top_k=3)

        self.assertGreaterEqual(len(results), 1)
        self.assertIn("건설기계", results[0]["decision_title"] + results[0]["industry"])
        self.assertIn("top_chunks", results[0])
        self.assertGreaterEqual(len(results[0]["top_chunks"]), 1)

    def test_risk_check_includes_disclaimer(self):
        result = self.service.risk_check("가맹점 식자재 구매 강제", top_k=3)

        self.assertIn(result["risk_level"], {"low", "medium", "high"})
        self.assertIn("matched_decisions", result)
        self.assertIn("법률 위반 확정 판단이 아닙니다", result["disclaimer"])


if __name__ == "__main__":
    unittest.main()

