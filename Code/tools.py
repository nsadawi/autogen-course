import requests
import sqlite3
import subprocess

def web_search(query: str, top_k: int = 5) -> str:
    """Search the web for the query and return top results as a short list."""
    # Replace with your search provider (SerpAPI, Bing, internal search, etc.)
    # Keep output compact and structured.
    results = [f"{i+1}. (mock) Result title for '{query}' - snippet..." for i in range(top_k)]
    return "\n".join(results)


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
