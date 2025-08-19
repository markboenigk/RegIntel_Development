#!/usr/bin/env python3
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    import index  # Import after dotenv
    query = os.environ.get("TEST_QUERY", "What news about Stryker?")

    print(f"Query: {query}")
    # Fetch top sources
    sources = await index.search_similar_documents(query, collection_name="rss_feeds", top_k=5)
    print(f"Sources fetched: {len(sources)}")
    if sources:
        first = sources[0]
        md = first.get('metadata', {})
        print("Top source:", md.get('article_title') or md.get('company_name'), md.get('published_date') or md.get('letter_date'))

    # Call chat with empty history
    response = await index.chat_with_gpt(query, [], sources)
    print("\n--- Chat Response ---\n")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())