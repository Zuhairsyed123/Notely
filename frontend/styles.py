"""Custom CSS styling for Notely Streamlit UI."""

CUSTOM_CSS = """
<style>
/* Main typography and container spacing */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Header & Brand */
.notely-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}

.notely-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1e1b4b;
    margin: 0;
    letter-spacing: -0.02em;
}

.notely-tagline {
    font-size: 1.05rem;
    color: #6366f1;
    font-weight: 500;
    margin-bottom: 24px;
}

/* Stat Cards */
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    margin-bottom: 12px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stat-card:hover {
    border-color: #c7d2fe;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #312e81;
    line-height: 1.2;
}

.stat-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-top: 4px;
}

/* Badges and Pills */
.badge-mock {
    background: #fef3c7;
    color: #92400e;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid #fde68a;
    display: inline-block;
}

.badge-ai {
    background: #ede9fe;
    color: #4338ca;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid #ddd6fe;
    display: inline-block;
}

.badge-extractive {
    background: #e0f2fe;
    color: #0369a1;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid #bae6fd;
    display: inline-block;
}

/* Citations & Excerpts */
.citation-box {
    background: #f8fafc;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #334155;
}

.citation-header {
    font-weight: 600;
    color: #4338ca;
    font-size: 0.82rem;
    margin-bottom: 4px;
}

/* Flashcard Container */
.flashcard {
    background: #ffffff;
    border: 2px solid #e0e7ff;
    border-radius: 16px;
    padding: 36px 32px;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.06);
    margin: 16px 0;
    position: relative;
}

.flashcard-question {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1e1b4b;
    line-height: 1.5;
}

.flashcard-answer {
    font-size: 1.15rem;
    font-weight: 500;
    color: #312e81;
    line-height: 1.6;
    background: #f5f3ff;
    border-radius: 8px;
    padding: 16px 20px;
    width: 100%;
}

.flashcard-source {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 14px;
}

/* Chunk Explorer Box */
.chunk-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
}

.chunk-meta {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 8px;
}

/* Clean Button Styling */
div.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease-in-out;
}

div.stButton > button:hover {
    border-color: #6366f1;
    color: #4f46e5;
}
</style>
"""
