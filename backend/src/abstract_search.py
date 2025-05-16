import arxiv

def get_abstracts_by_keywords(keywords, max_results=15):
    keywords = [kw.strip() for kw in keywords.split(',')]

    search_query = f"{keywords[0]} AND {keywords[1]}"

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client()
    abstracts = []

    for result in client.results(search):
        abstracts.append(result.summary)

    return abstracts