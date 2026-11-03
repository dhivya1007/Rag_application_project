
import streamlit as st


# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="News Search & Summarization",
    page_icon="📰",
    layout="wide"
)

# Display the UI immediately
st.title("📰 News Search & Summarization")

st.write(
    "Search news articles using hybrid keyword and semantic retrieval, "
    "then generate an AI-powered summary."
)


# -------------------------------
# Search form
# -------------------------------

with st.form("search_form"):

    query = st.text_input(
        "Enter your news query",
        placeholder="Example: Minneapolis bridge collapse"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        n = st.slider(
            "Number of articles",
            min_value=1,
            max_value=10,
            value=5
        )

    with col2:
        summary_length = st.selectbox(
            "Summary length",
            ["short", "medium", "long"],
            index=1
        )

    with col3:
        use_rerank = st.checkbox(
            "Enable Cross-Encoder reranking",
            value=True
        )

    submitted = st.form_submit_button(
        "🔍 Search and Summarize",
        use_container_width=True
    )


# -------------------------------
# Search and summarization
# -------------------------------

if submitted:

    if not query.strip():
        st.warning("Please enter a search query.")

    else:

        try:
            # Import only when search is requested
            st.info("Loading search and summarization modules...")

            from hybrid_search import search_articles
            from summarizer import generate_summary

            # Retrieve articles
            with st.spinner("Searching news articles..."):

                articles = search_articles(
                    query=query,
                    n=n,
                    use_rerank=use_rerank
                )

            if not articles:

                st.warning("No relevant articles found.")

            else:

                st.success(
                    f"Retrieved {len(articles)} relevant articles."
                )

                # Display retrieved articles
                st.subheader("Retrieved Articles")

                for i, article in enumerate(articles, start=1):

                    with st.expander(
                        f"{i}. {article['title']}"
                    ):

                        st.write(
                            "**Location:**",
                            article["location"] or "Not available"
                        )

                        st.write(
                            "**Retrieval method:**",
                            article["method"]
                        )

                        st.write(
                            "**Hybrid score:**",
                            round(article["rrf_score"], 4)
                        )

                        if article["rerank_score"] is not None:

                            st.write(
                                "**Reranker score:**",
                                round(article["rerank_score"], 4)
                            )

                        st.write("**Article excerpt:**")
                        st.write(article["text"])

                # Generate Gemini summary
                st.subheader("AI-Generated Answer")

                with st.spinner("Gemini is generating your summary..."):

                    summary = generate_summary(
                        articles=articles,
                        query=query,
                        length=summary_length
                    )

                st.markdown(summary)

        except Exception as e:

            st.error(f"Something went wrong: {e}")
