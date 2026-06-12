import unittest

from statute_retriever import (
    StatuteRetrievalService,
    StatuteSearchConfig,
    format_statute_result,
    infer_jo_number,
    rrf_score,
    tokenize_korean,
)


class FakeBM25:
    def get_scores(self, tokens):
        return [0.0, 3.0, 1.0]


class StatuteRetrievalTest(unittest.TestCase):
    def test_rrf_score_uses_rank_and_k(self):
        self.assertAlmostEqual(rrf_score(1, 60), 1 / 61)

    def test_bm25_search_keeps_positive_scores_in_rank_order(self):
        service = StatuteRetrievalService.__new__(StatuteRetrievalService)
        service.bm25 = FakeBM25()
        service.bm25_ids = ["a", "b", "c"]

        self.assertEqual(service._bm25_search("하도급 대금", 3), ["b", "c"])

    def test_rrf_fuse_combines_dense_and_bm25(self):
        service = StatuteRetrievalService.__new__(StatuteRetrievalService)
        scores, dense_ranks, bm25_ranks = service._rrf_fuse(
            ["a", "b"],
            ["b", "c"],
            StatuteSearchConfig(rrf_k=60),
        )

        self.assertGreater(scores["b"], scores["a"])
        self.assertEqual(dense_ranks["a"], 1)
        self.assertEqual(bm25_ranks["c"], 2)

    def test_format_result_exposes_statute_fields(self):
        result = format_statute_result(
            {
                "id": "LAW-하도급법-1조",
                "document": "법률명: 하도급거래 공정화에 관한 법률\n본문",
                "metadata": {
                    "statute_id": "LAW-하도급법-1조",
                    "law_title": "하도급거래 공정화에 관한 법률",
                    "jo_number": "1조",
                    "jo_title": "목적",
                    "doc_type": "statute",
                },
            },
            {"LAW-하도급법-1조": 0.25},
            {"LAW-하도급법-1조": 1},
            {},
        )

        self.assertEqual(result["statute_id"], "LAW-하도급법-1조")
        self.assertEqual(result["law_title"], "하도급거래 공정화에 관한 법률")
        self.assertEqual(result["score"], 0.25)
        self.assertEqual(result["dense_rank"], 1)

    def test_infer_jo_number(self):
        self.assertEqual(infer_jo_number("하도급거래 공정화에 관한 법률 제13조"), "13조")

    def test_tokenize_korean_has_fallback_tokens(self):
        self.assertTrue(tokenize_korean("하도급 대금 지급"))


if __name__ == "__main__":
    unittest.main()
