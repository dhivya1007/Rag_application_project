import os
import time
import argparse
from groq import Groq
from dotenv import load_dotenv
from hybrid_search import search_articles

# Load environment variables
load_dotenv()

# Initialize the Groq Client correctly
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_summary(articles, query, length="medium"):
    if not articles:
        return "No relevant articles found."

    # Prepare article context
    context = ""
    for i, article in enumerate(articles, start=1):
        context += f"""
        Article {i}
        Title: {article.get('title', 'Untitled')}
        Location: {article.get('location', 'Unknown')}

        Content:
        {article.get('text', '')}
        """

    # Summary length instructions
    length_instructions = {
        "short": "Write a concise summary in 3-4 sentences.",
        "medium": "Write a detailed summary in 2-3 paragraphs.",
        "long": "Write a comprehensive summary with key facts and details."
    }

    # For chat/completion endpoints, separation of system instructions and user context works best
    system_instruction = """
    You are a factual news summarization assistant.
    Instructions:
    1. Use only information from the retrieved articles.
    2. Do not invent facts or use outside knowledge.
    3. Ignore information unrelated to the query.
    4. If information is insufficient, clearly say so.
    5. Mention conflicting information if present.
    6. Identify the article titles supporting the summary.
    """

    user_prompt = f"""
    User query: {query}

    Retrieved articles:
    {context}

    Summary length:
    {length_instructions.get(length, length_instructions["medium"])}
    """

    # Call Groq API with robust error handling
    for attempt in range(3):
        try:
            # Corrected to use Groq's Chat Completion structure
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",  # Fast, highly accurate open model on Groq
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2  # Keep temperature low for deterministic RAG results
            )

            # Extract content from Groq's return payload structure
            return response.choices[0].message.content

        except Exception as e:
            if attempt < 2:
                print(f"Groq API connection hiccup. Retrying... Attempt {attempt + 2}")
                time.sleep(2 * (attempt + 1))
            else:
                raise

# -----------------------------------
# TERMINAL TESTING INTERFACE
# -----------------------------------
# --- TERMINAL TESTING INTERFACE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the Groq RAG Summarizer via Terminal."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The search or user query to summarize against."
    )
    parser.add_argument(
        "--length",
        type=str,
        choices=["short", "medium", "long"],
        default="medium",
        help="Length of the summary."
    )

    args = parser.parse_args()

    print("\n--- Fetching Live Data ---")
    print(f"Query: {args.query}")
    print(f"Length: {args.length}\n")

    # 1. Fetch live articles from your local database/search module using the query
    try:
        live_articles = search_articles(args.query)
    except Exception as e:
        print(f"Error fetching articles from hybrid_search: {e}")
        live_articles = []

    # 2. Run the summarizer on the actual fetched data
    summary_result = generate_summary(
        live_articles,
        args.query,
        args.length
    )

    print("--- Model Response ---")
    print(summary_result)
