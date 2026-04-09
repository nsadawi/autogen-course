import requests
import sqlite3
import subprocess

#def web_search(query: str, top_k: int = 5) -> str:
#    """Search the web for the query and return top results as a short list."""
#    # Replace with your search provider (SerpAPI, Bing, internal search, etc.)
#    # Keep output compact and structured.
#    results = [f"{i+1}. (mock) Result title for '{query}' - snippet..." for i in range(top_k)]
#    return "\n".join(results)

# ---- REAL WEB SEARCH TOOL (SerpAPI) ----
def web_search(query: str, top_k: int = 5) -> str:
    """
    Perform a real web search using SerpAPI (Google Search).
    Returns a compact, structured summary suitable for LLM reasoning.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "ERROR: SERPAPI_API_KEY not set."

    params = {
        "engine": "google",
        "q": query,
        "num": top_k,
        "api_key": api_key,
    }

    try:
        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        organic = data.get("organic_results", [])[:top_k]
        if not organic:
            return "No relevant search results found."

        results = []
        for i, item in enumerate(organic, start=1):
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No snippet")
            link = item.get("link", "No link")
            results.append(
                f"{i}. {title}\n"
                f"   {snippet}\n"
                f"   Source: {link}"
            )

        return "\n".join(results)

    except Exception as e:
        return f"Web search failed: {str(e)}"

def db_get_order(order_id: str) -> str:
    """Lookup an order by ID and return status, date, and last update."""
    con = sqlite3.connect("support.db")
    cur = con.cursor()
    cur.execute("SELECT status, updated_at FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return f"No order found for order_id={order_id}"
    status, updated_at = row
    return f"order_id={order_id} status={status} updated_at={updated_at}"


def run_tests(command: str = "pytest -q") -> str:
    """Run the test suite and return the output (stdout + stderr)."""
    # In production: allowlist, timeout, no shell=True, containerize, etc.
    completed = subprocess.run(
        command.split(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return f"exit_code={completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
