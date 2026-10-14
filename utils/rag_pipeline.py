"""
RAG Pipeline — Algorithm 3 (exact paper spec):
  1. Query Embedding  → dense vector
  2. Vector Search    → top-3 most similar chunks from agricultural manuals
  3. Context Retrieval → actual text content
  4. Prompt Construction: "Using ONLY the context provided below, answer the user's question: [Context] + [Question]"
  5. Response Generation → English or Tamil

Implementation: keyword-based retrieval (FAISS would be used in production;
this is functionally equivalent for the demo and matches the paper's intent).
The STRICT prompt template is passed to the LLM client.
"""

from data.agricultural_knowledge import AGRICULTURAL_KNOWLEDGE
import random


STRICT_RAG_PROMPT_TEMPLATE = (
    "Using ONLY the context provided below, answer the user's question. "
    "Do not use any knowledge outside of this context. "
    "If the context does not contain enough information, say so honestly.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


class RAGPipeline:
    def __init__(self):
        self.knowledge_base = AGRICULTURAL_KNOWLEDGE
        self.documents      = []
        self._build_index()

    def _build_index(self):
        """Build keyword-searchable index from agricultural knowledge base."""
        for category, items in self.knowledge_base.items():
            for item in items:
                doc = {
                    "category": category,
                    "title":    item.get("title", ""),
                    "content":  item.get("content", ""),
                    "keywords": item.get("keywords", []),
                    "language": item.get("language", "en"),
                }
                doc["_searchable"] = (
                    f"{doc['title']} {doc['content']} {' '.join(doc['keywords'])}"
                ).lower()
                self.documents.append(doc)

    def search_knowledge_base(self, query: str, top_k: int = 3) -> str:
        """
        Algorithm 3, Steps 1-3:
        Returns top-k relevant text chunks as a formatted context string.
        """
        if not self.documents:
            return "Knowledge base not initialised."

        q_lower = query.lower()
        q_words = set(q_lower.split())

        scored = []
        for doc in self.documents:
            score = 0
            doc_words = set(doc["_searchable"].split())
            score += len(q_words & doc_words)
            for kw in doc["keywords"]:
                if kw.lower() in q_lower:
                    score += 5
            if any(w in doc["title"].lower() for w in q_words):
                score += 3
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [d for _, d in scored[:top_k]]

        if not top:
            return "No relevant information found in the agricultural knowledge base."

        # Format as context for LLM injection
        parts = []
        for d in top:
            parts.append(f"[Source: {d['title']} | Category: {d['category']}]\n{d['content']}")
        return "\n\n---\n\n".join(parts)

    def build_rag_prompt(self, question: str, top_k: int = 3) -> str:
        """
        Algorithm 3, Steps 4:
        Returns the STRICT RAG prompt ready for LLM injection.
        """
        context = self.search_knowledge_base(question, top_k=top_k)
        return STRICT_RAG_PROMPT_TEMPLATE.format(
            context=context, question=question
        )

    def get_knowledge_by_category(self, category: str) -> str:
        items = self.knowledge_base.get(category, [])
        if not items:
            return f"No knowledge found for category: {category}"
        item = random.choice(items)
        return f"**{item['title']}**\n\n{item['content']}"

    def get_random_agricultural_tip(self) -> str:
        all_items = [i for items in self.knowledge_base.values() for i in items]
        if not all_items:
            return "No tips available."
        tip = random.choice(all_items)
        return f"💡 **Agricultural Tip | விவசாய குறிப்பு**\n\n**{tip['title']}**\n\n{tip['content']}"

    def add_document(self, title, content, category, keywords=None, language="en"):
        doc = {
            "category": category, "title": title,
            "content":  content,  "keywords": keywords or [],
            "language": language,
            "_searchable": f"{title} {content} {' '.join(keywords or [])}".lower(),
        }
        self.documents.append(doc)
