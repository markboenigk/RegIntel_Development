#!/usr/bin/env python3
"""
Debug script to see exactly what context is being built for FDA warning letters
"""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def debug_fda_context():
    """Debug the context building for FDA warning letters"""
    
    import index
    
    # Test query
    query = "What is the warning letter from Artisan Vapor Company about?"
    
    print("🔍 Debugging FDA Warning Letters Context Building")
    print("=" * 50)
    print(f"Query: {query}")
    
    try:
        # Get sources
        sources = await index.search_similar_documents(query, collection_name="fda_warning_letters", top_k=5)
        print(f"\n📚 Sources found: {len(sources)}")
        
        if sources:
            first_source = sources[0]
            print(f"\n🔍 First source details:")
            print(f"  Title: {first_source.get('title', 'No title')}")
            print(f"  Content length: {len(first_source.get('content', ''))}")
            print(f"  Collection: {first_source.get('collection', 'No collection')}")
            
            metadata = first_source.get('metadata', {})
            print(f"  Company: {metadata.get('company_name', 'No company')}")
            print(f"  Date: {metadata.get('letter_date', 'No date')}")
            
            # Show first 200 chars of content
            content = first_source.get('content', '')
            print(f"\n📄 Content preview (first 200 chars):")
            print(f"  {content[:200]}...")
            
            # Show the exact context that would be built
            print(f"\n🔧 Building context manually...")
            
            collection_type = first_source.get('collection', 'general')
            context = f"\n\nRelevant source from {collection_type.replace('_', ' ').title()}:\n"
            
            metadata = first_source.get('metadata', {})
            if collection_type == "fda_warning_letters":
                title = metadata.get('company_name', 'Unknown Company')
                company = metadata.get('company_name', 'Unknown Company')
                date = metadata.get('letter_date', 'Unknown Date')
                context += f"1. {title} - Company: {company}, Date: {date}\n"
            else:
                title = metadata.get('article_title', 'Unknown Title')
                feed = metadata.get('feed_name', 'Unknown Feed')
                date = metadata.get('published_date', 'Unknown Date')
                context += f"1. {title} - Feed: {feed}, Date: {date}\n"
            
            # Use content field and show what would be sent
            content = first_source.get('content', '')[:1200]
            context += f"   {content}...\n\n"
            
            print(f"\n📋 Final context that would be sent to AI:")
            print(f"  Length: {len(context)} characters")
            print(f"  Preview: {context[:500]}...")
            
        else:
            print("❌ No sources found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_fda_context()) 