from retriever import FaircheckRetriever, RetrieverConfig


def print_results(title, results, rank_maps):
    print(f"\n[{title}]")

    for i, (doc_id, doc, meta, score) in enumerate(results, start=1):
        dense_rank = rank_maps["dense"].get(doc_id, "-")
        bm25_rank = rank_maps["bm25"].get(doc_id, "-")
        rrf = rank_maps["rrf"].get(doc_id, 0)

        print(
            f"\n{i}위 "
            f"(rerank: {score:.4f} | rrf: {rrf:.4f} | "
            f"dense: {dense_rank}위 | bm25: {bm25_rank}위)"
        )
        print(f"  ID: {doc_id}")
        print(f"  제목: {meta.get('의결서제목', '')}")
        print(f"  위반유형: {meta.get('위반유형', '')}")
        print(f"  업종: {meta.get('업종', '')}")
        print(f"  내용: {doc[:120]}...")


def print_document_contexts(doc_contexts, rank_maps):
    print("\n[문서별 RAG context 후보]")

    for i, context in enumerate(doc_contexts, start=1):
        meta = context["meta"]
        print(f"\n{i}위 문서 (best rerank: {context['best_score']:.4f})")
        print(f"  대표 ID: {context['best_doc_id']}")
        print(f"  제목: {meta.get('의결서제목', '')}")
        print(f"  위반유형: {meta.get('위반유형', '')}")
        print(f"  업종: {meta.get('업종', '')}")

        for j, chunk in enumerate(context["chunks"], start=1):
            doc_id = chunk["doc_id"]
            dense_rank = rank_maps["dense"].get(doc_id, "-")
            bm25_rank = rank_maps["bm25"].get(doc_id, "-")
            rrf = rank_maps["rrf"].get(doc_id, 0)

            print(
                f"    chunk {j}: {doc_id} "
                f"(rerank: {chunk['score']:.4f} | rrf: {rrf:.4f} | "
                f"dense: {dense_rank}위 | bm25: {bm25_rank}위)"
            )
            print(f"      내용: {chunk['doc'][:120]}...")


if __name__ == "__main__":
    config = RetrieverConfig()
    retriever = FaircheckRetriever(config)

    queries = [
        "배달앱 수수료 불공정거래",
        "병원 의약품 리베이트",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"질의: {query}")
        print(f"{'=' * 60}")

        result = retriever.search(query)

        if result["search_query"] != result["query"]:
            print(f"검색 질의 재작성: {result['search_query']}")

        rank_maps = {
            "rrf": result["rrf_scores"],
            "dense": result["dense_rank_map"],
            "bm25": result["bm25_rank_map"],
        }

        print_results("Chunk 단위 결과", result["chunk_results"], rank_maps)
        print_results("문서 단위 결과", result["doc_results"], rank_maps)
        print_document_contexts(result["doc_contexts"], rank_maps)
