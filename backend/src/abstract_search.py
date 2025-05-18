import arxiv

def get_abstracts_by_keywords(keywords, max_results=15):
    """
    Fetches the abstracts of research papers from arXiv based on a list of provided keywords.

    Args:
        keywords (list): A list of keywords (strings) to search for in the arXiv database.
        max_results (int, optional): The maximum number of search results to return. Default is 15.

    Raises:
        ValueError: If no keywords are provided.

    Returns:
        dict: A dictionary where the keys are the titles of the papers and the values are dictionaries
              containing the abstract ("abstract") and the URL ("url") of each paper.
    """
    if len(keywords) == 0:
        raise ValueError("At least one keyword must be provided.")
    else:
        keywords = keywords[:3]
        search_query = " AND ".join(f'"{kw.strip()}"' for kw in keywords)

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client()
    results = {}

    for result in client.results(search):
        results[result.title] = {
            "abstract": result.summary,
            "url": result.entry_id
        }

    return results
