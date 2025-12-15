"""
LangGraph-based multi-modal travel assistant with manual tool execution, vector
lookup, and mock API integrations. Built to satisfy the AI_Engineer
requirements and the provided execution plan.
"""

import hashlib
import json
import operator
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None  # type: ignore
try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None  # type: ignore


load_dotenv(override=False)

# --------------------------
# Embeddings + Vector Store
# --------------------------


class HashEmbedder:
    """Deterministic bag-of-words style embedder to avoid network calls."""

    def __init__(self, dim: int = 96) -> None:
        self.dim = dim

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[A-Za-z']+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            for idx in digest[:6]:
                vector[idx % self.dim] += 1.0
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm

    def embed_query(self, text: str) -> List[float]:
        return self._encode(text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode(text).tolist() for text in texts]


@dataclass
class CityDoc:
    city: str
    country: str
    summary: str


class CityVectorStore:
    """Minimal FAISS-backed semantic lookup for the seeded city docs."""

    def __init__(self, docs: List[CityDoc], embedder: Optional[Any] = None, dim: Optional[int] = None) -> None:
        self.docs = docs
        base_dim = dim or 96
        self.embedder = embedder or HashEmbedder(base_dim)
        summaries = [doc.summary for doc in docs]
        try:
            emb_vectors = self.embedder.embed_documents(summaries)
        except Exception:
            self.embedder = HashEmbedder(base_dim)
            emb_vectors = self.embedder.embed_documents(summaries)
        embeddings = np.array(emb_vectors, dtype="float32")
        if embeddings.size == 0 or embeddings.ndim != 2:
            self.embedder = HashEmbedder(base_dim)
            embeddings = np.array(self.embedder.embed_documents(summaries), dtype="float32")
        infer_dim = embeddings.shape[1]
        self.dim = infer_dim
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(infer_dim)
        self.index.add(embeddings)

    def similarity_search(self, query: str, threshold: Optional[float] = None) -> Optional[Tuple[CityDoc, float]]:
        score_threshold = threshold
        if score_threshold is None:
            try:
                score_threshold = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.75"))
            except Exception:
                score_threshold = 0.75
        query_vec = np.array([self.embedder.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, 1)
        score = float(scores[0][0])
        idx = int(indices[0][0])
        if idx < 0 or (score_threshold is not None and score < score_threshold):
            return None
        return self.docs[idx], score


def seed_city_docs() -> List[CityDoc]:
    return [
        CityDoc(
            city="Paris",
            country="France",
            summary=(
                "Paris blends grand boulevards, cafe culture, and contemporary art. "
                "Anchor points include the Seine, Louvre, Pompidou, and the Right/Left Bank split. "
                "Food spans bistros to patisserie classics, while day trips to Versailles or Champagne are easy."
            ),
        ),
        CityDoc(
            city="Tokyo",
            country="Japan",
            summary=(
                "Tokyo pairs neon Shibuya crossings with quiet shrines. "
                "Neighborhoods like Shinjuku, Asakusa, and Daikanyama offer distinct vibes. "
                "Cuisine ranges from ramen alleys to Michelin sushi, with Shinkansen access to Hakone and Kyoto."
            ),
        ),
        CityDoc(
            city="New York",
            country="USA",
            summary=(
                "New York City stacks skyline views, Central Park green space, and diverse borough food scenes. "
                "Museums (MoMA, Met), Broadway, and Hudson River walks anchor days; subway makes quick hops across Manhattan and Brooklyn."
            ),
        ),
    ]


def _make_embedder() -> Any:
    """Prefer local Ollama embeddings; fallback to deterministic hash if unavailable."""
    
    return OllamaEmbeddings(model="nomic-embed-text")


CITY_VECTOR_STORE = CityVectorStore(seed_city_docs(), embedder=_make_embedder())


# --------------------------
# Mock external tools
# --------------------------



def _live_search_city(city: str) -> Optional[str]:
    try:
        with DDGS() as ddg:
            results = ddg.text(f"{city} travel guide landmarks food vibe", max_results=5)
        snippets = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            for key in ("body", "snippet", "title"):
                if key in item and isinstance(item[key], str):
                    snippets.append(item[key])
                    break
        if snippets:
            joined = ". ".join(snippets[:3])
            return f"{city}: {joined}"
    except Exception as exc:
        print(f"[search] live search failed: {exc}")
    return None


def search_city_info(city: str) -> Tuple[str, str]:
    """
    Generate a travel summary via live search (if enabled) or LLM with retries.
    Returns (summary, source_tag).
    """
    use_live = os.getenv("USE_LIVE_SEARCH", "0") == "1"
    if use_live:
        result = _with_retries(lambda: _live_search_city(city), attempts=3, delay=0.8)
        if result:
            return result, "live_search"
        print("[search] live search failed after retries; falling back to LLM.")

    if ChatOllama is not None:
        def _call():
            llm = ChatOllama(model=os.getenv("OLLAMA_SEARCH_MODEL", "llama3.2"), temperature=0.3)
            resp = llm.invoke(
                f"Provide a crisp 3-sentence travel summary for {city}. Focus on landmarks, food, vibe, and nearby day trips. "
                f"Do not start your response with 'I couldn't find' or 'I don't have information'. Start directly with the travel summary."
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        llm_summary = _with_retries(_call, attempts=3, delay=0.6)
        if llm_summary:
            return llm_summary, "llm_search"
        print("[search] LLM search failed after retries; using template fallback.")

    return (
        f"{city} blends notable landmarks, local markets, and food culture. Day trips and neighborhoods offer variety even without live search data.",
        "template_search",
    )


def get_weather_forecast(city: str, start_offset: int = 0, days: int = 6) -> List[Dict[str, Any]]:
    """Mock weather API returning structured multi-day data."""
    random.seed(f"{city}-{start_offset}-{days}")
    today = datetime.utcnow() + timedelta(days=start_offset)
    base_temp = random.randint(8, 22)
    forecast = []
    for idx in range(days):
        date = today + timedelta(days=idx)
        temp = base_temp + random.randint(-2, 6)
        forecast.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "temperature_c": temp,
                "condition": random.choice(["sunny", "cloudy", "showers", "windy", "clear"]),
            }
        )
    time.sleep(0.2)  # simulate latency
    return forecast


def get_city_images(city: str, count: int = 4) -> List[str]:
    """Return Unsplash direct links; fallback to static curated URLs."""
    def _unsplash() -> List[str]:
        access_key = os.getenv("UNSPLASH_ACCESS_KEY")
        if not access_key:
            return []
        try:
            resp = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": city, "per_page": count, "orientation": "landscape"}, # Added orientation for better UI
                headers={"Authorization": f"Client-ID {access_key}"},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                return [item["urls"]["regular"] for item in data.get("results", [])]
            else:
                return []
        except Exception as e:
            return []

    curated = {
        "paris": [
            "https://images.unsplash.com/photo-1502602898657-3e91760cbb34",
            "https://images.unsplash.com/photo-1467269204594-9661b134dd2b",
        ],
        "tokyo": [
            "https://images.unsplash.com/photo-1505069442581-7e7c7b9dc2f2",
            "https://images.unsplash.com/photo-1498654896293-37aacf113fd9",
        ],
        "new york": [
            "https://images.unsplash.com/photo-1469474968028-56623f02e42e",
            "https://images.unsplash.com/photo-1488747279002-c8523379faaa",
        ],
    }

    urls = _unsplash()
    if not urls:
        urls = curated.get(city.lower(), [])[:count]

    if len(urls) < count:
        filler = [
            "https://images.unsplash.com/photo-1505764706515-aa95265c5abc",
            "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee",
            "https://images.unsplash.com/photo-1491553895911-0055eca6402d",
        ]
        urls.extend(filler[: max(0, count - len(urls))])
    return [f"{url}?auto=format&fit=crop&w=1200&q=80" for url in urls]


# --------------------------
# State + helpers
# --------------------------


class TravelState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    city: Optional[str]
    city_summary: Optional[str]
    weather_forecast: List[Dict[str, Any]]
    image_urls: List[str]
    timeframe: Dict[str, Any]
    needs_images: bool
    needs_weather: bool
    pending_search: bool
    source: Optional[str]
    final_response: Optional[Dict[str, Any]]


def _extract_city(text: str, fallback: Optional[str]) -> Optional[str]:
    known = [doc.city for doc in CITY_VECTOR_STORE.docs]
    lowered = text.lower()
    for candidate in known:
        if candidate.lower() in lowered:
            return candidate
    return fallback


def _parse_timeframe(text: str, previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    lowered = text.lower()
    days = 6
    offset = 0
    label = "this week"
    explicit = re.search(r"(\d+)\s*(day|days)", lowered)
    if "next week" in lowered:
        offset, days, label = 7, 7, "next week"
    elif "tomorrow" in lowered:
        offset, days, label = 1, 3, "tomorrow"
    elif "today" in lowered:
        offset, days, label = 0, 1, "today"
    elif explicit:
        days = max(1, int(explicit.group(1)))
        label = f"next {days} days"
    elif previous:
        return previous
    return {"offset": offset, "days": days, "label": label}


def _make_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "args": args,
        "id": f"{name}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
    }


def _with_retries(fn, attempts: int = 2, delay: float = 0.5):
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            print(f"[retryable] attempt {attempt + 1}/{attempts} failed: {exc}")
            if attempt < attempts - 1:
                time.sleep(delay)
    return None


def _llm_extract_city(text: str, fallback: Optional[str]) -> Optional[str]:
    """Ask Llama to extract city name robustly; fall back to heuristic if unavailable."""
    if ChatOllama is None or os.getenv("USE_LLM_CITY", "1") != "1":
        return None
    system_prompt = (
        "You extract city names from user travel requests.\n"
        "Rules:\n"
        "1) If the user mentions a city, return ONLY the city name (retain multi-word, e.g., 'New York').\n"
        "2) If no city is mentioned, return the previous city if provided, else return 'UNKNOWN'.\n"
        "3) Ignore time-related words like week, tomorrow, today, next, previous.\n"
        "Respond as JSON: {\"city\": \"<name|UNKNOWN>\"}."
        " Do not start responses with apologies; only return the JSON."
    )

    def _call():
        llm = ChatOllama(model=os.getenv("OLLAMA_CITY_MODEL", "llama3.2"), temperature=0)
        return llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({"text": text, "previous_city": fallback})},
            ]
        )

    resp = _with_retries(_call, attempts=3, delay=0.6)
    if not resp:
        return None
    content = resp.content if hasattr(resp, "content") else str(resp)
    parsed = None
    for candidate in [content, content.strip().strip("`")]:
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            continue
    if isinstance(parsed, dict):
        city = parsed.get("city")
        if isinstance(city, str) and city and city.upper() != "UNKNOWN":
            return city.strip()
    return None


def _llm_plan_tools(
    city: Optional[str],
    timeframe: Dict[str, Any],
    needs_weather: bool,
    needs_images: bool,
) -> List[Dict[str, Any]]:
    """Let the LLM decide which tools to call; fallback to deterministic plan."""
    planned: List[Dict[str, Any]] = []
    if not city:
        return planned

    if ChatOllama is not None and os.getenv("USE_LLM_PLANNER", "1") == "1":
        prompt = (
            "You are a tool planner for a travel assistant.\n"
            "Tools:\n"
            "- get_weather_forecast(city, start_offset, days)\n"
            "- get_city_images(city)\n"
            f"City: {city}\n"
            f"Timeframe: offset={timeframe.get('offset', 0)}, days={timeframe.get('days', 6)}, label={timeframe.get('label', 'this week')}\n"
            f"Weather needed: {needs_weather}, Images needed: {needs_images}\n"
            "Return ONLY JSON list of tool calls. Example:\n"
            '[{"name":"get_weather_forecast","args":{"city":"Paris","start_offset":0,"days":6}}, {"name":"get_city_images","args":{"city":"Paris"}}]'
        )
        try:
            llm = ChatOllama(model=os.getenv("OLLAMA_PLANNER_MODEL", "llama3.2"), temperature=0)
            resp = llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            parsed = None
            for candidate in [raw, raw.strip().strip("`")]:
                try:
                    parsed = json.loads(candidate)
                    break
                except Exception:
                    continue
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    args = item.get("args") or {}
                    if name not in {"get_weather_forecast", "get_city_images"}:
                        continue
                    args.setdefault("city", city)
                    if name == "get_weather_forecast":
                        args.setdefault("start_offset", timeframe.get("offset", 0))
                        args.setdefault("days", timeframe.get("days", 6))
                    planned.append({"name": name, "args": args})
        except Exception:
            planned = []

    if not planned:
        if needs_weather:
            planned.append(
                {
                    "name": "get_weather_forecast",
                    "args": {"city": city, "start_offset": timeframe.get("offset", 0), "days": timeframe.get("days", 6)},
                }
            )
        if needs_images:
            planned.append({"name": "get_city_images", "args": {"city": city}})
    return planned


def _llm_router_decide(city: str, score: float, user_text: str) -> bool:
    """Ask the LLM whether to bypass vector hit and search anyway."""
    enabled = os.getenv("USE_LLM_ROUTER", "1") == "1" and ChatOllama is not None
    if not enabled:
        return False
    prompt = (
        "You are a routing controller for a travel assistant.\n"
        "Decide if we should perform a web search given the following:\n"
        f"- City: {city}\n"
        f"- Vector similarity score (0-1): {score:.2f}\n"
        f"- User text: {user_text}\n"
        "Return ONLY one word: SEARCH (if you want a web search) or LOCAL (if vector data is enough).\n"
        "Prefer SEARCH if the score is <0.50 or the user hints at recent updates, news, or events."
    )
    try:
        llm = ChatOllama(model=os.getenv("OLLAMA_ROUTER_MODEL", "llama3.2"), temperature=0)
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return "SEARCH" in text.upper()
    except Exception:
        return False


def _maybe_refine_summary(city: str, summary: Optional[str], forecast: List[Dict[str, Any]], timeframe: Dict[str, Any]) -> Optional[str]:
    """Optionally ask an Ollama chat model to reframe the summary; gated by USE_LLM_SUMMARY."""
    if not summary or os.getenv("USE_LLM_SUMMARY", "0") != "1":
        return summary
    if ChatOllama is None:
        return summary
    try:
        llm = ChatOllama(model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2"), temperature=0.3)
        prompt = (
            f"Rewrite this city overview for {city} in 3-4 sentences. "
            f"Emphasize experiences relevant to the timeframe '{timeframe.get('label', 'this week')}'. "
            f"Base summary: {summary}. "
            f"Weather snapshot: {forecast}."
        )
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        return summary


# --------------------------
# LangGraph nodes
# --------------------------


def city_info_node(state: TravelState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    latest_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not latest_human:
        return {}

    user_text = latest_human.content
    previous_city = state.get("city")
    # Prefer LLM extraction; fallback to heuristic parser.
    candidate_city = _llm_extract_city(user_text, previous_city) or _extract_city(user_text, previous_city)

    # Validate candidate via vector-store similarity when a previous city exists.
    city = candidate_city
    if candidate_city and previous_city and candidate_city != previous_city:
        best_match = CITY_VECTOR_STORE.similarity_search(candidate_city, threshold=-1.0)
        low_conf_threshold = float(os.getenv("CITY_GUARD_THRESHOLD", "0.05"))
        score = best_match[1] if best_match else 0.0
        if score < low_conf_threshold:
            city = previous_city
    timeframe = _parse_timeframe(user_text, state.get("timeframe"))

    updates: Dict[str, Any] = {
        "city": city,
        "timeframe": timeframe,
        "needs_weather": True,
        "needs_images": False,
        "pending_search": False,
    }

    if not city:
        return updates

    new_city = city != state.get("city")
    reuse_summary = bool(state.get("city_summary")) and not new_city
    if reuse_summary:
        updates["needs_images"] = False
        updates["pending_search"] = False
        return updates

    vector_hit = CITY_VECTOR_STORE.similarity_search(city)
    if vector_hit:
        doc, score = vector_hit
        should_search = _llm_router_decide(city, score, user_text)
        if not should_search:
            updates.update(
                {
                    "city_summary": doc.summary,
                    "source": f"vector_store@{score:.2f}",
                    "pending_search": False,
                    "needs_images": True,
                }
            )
            return updates

    tool_call = _make_tool_call("search_city_info", {"city": city})
    ai_msg = AIMessage(content=f"Searching web for {city}", tool_calls=[tool_call])
    updates.update({"messages": [ai_msg], "pending_search": True, "needs_images": True})
    return updates


def route_after_city_info(state: TravelState) -> str:
    return "search_node" if state.get("pending_search") else "planner_node"


def search_node(state: TravelState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    ai_with_tools = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )
    if not ai_with_tools:
        return {"pending_search": False}

    results = []
    summary = state.get("city_summary")
    source = state.get("source")
    for call in ai_with_tools.tool_calls:
        if call["name"] != "search_city_info":
            continue
        city = call["args"].get("city")
        summary, source = search_city_info(city)
        results.append(
            ToolMessage(
                content=summary,
                tool_call_id=call["id"],
                name=call["name"],
            )
        )
    payload: Dict[str, Any] = {
        "messages": results,
        "city_summary": summary,
        "source": source,
        "pending_search": False,
    }
    return payload


def planner_node(state: TravelState) -> Dict[str, Any]:
    city = state.get("city")
    timeframe = state.get("timeframe", {"offset": 0, "days": 6, "label": "this week"})
    if not city:
        return {}
    needs_weather = state.get("needs_weather", True)
    needs_images = state.get("needs_images", False) or not state.get("image_urls")

    planned = _llm_plan_tools(city, timeframe, needs_weather, needs_images)
    if not planned:
        return {}

    tool_calls = [_make_tool_call(item["name"], item.get("args", {})) for item in planned]
    ai_msg = AIMessage(content="Planner decided tools to run", tool_calls=tool_calls)
    return {"messages": [ai_msg]}


def weather_node(state: TravelState) -> Dict[str, Any]:
    ai_with_tools = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )
    if not ai_with_tools:
        return {}

    forecast: List[Dict[str, Any]] = state.get("weather_forecast", [])
    tool_msgs: List[ToolMessage] = []
    for call in ai_with_tools.tool_calls:
        if call["name"] != "get_weather_forecast":
            continue
        args = call["args"]
        forecast = get_weather_forecast(
            args.get("city", state.get("city")),
            args.get("start_offset", 0),
            args.get("days", 6),
        )
        tool_msgs.append(
            ToolMessage(
                content=json.dumps(forecast),
                tool_call_id=call["id"],
                name=call["name"],
            )
        )
    return {"messages": tool_msgs, "weather_forecast": forecast}


def image_node(state: TravelState) -> Dict[str, Any]:
    ai_with_tools = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )
    if not ai_with_tools:
        return {}

    images: List[str] = state.get("image_urls", [])
    tool_msgs: List[ToolMessage] = []
    for call in ai_with_tools.tool_calls:
        if call["name"] != "get_city_images":
            continue
        args = call["args"]
        images = get_city_images(args.get("city", state.get("city")), count=6)
        tool_msgs.append(
            ToolMessage(
                content=json.dumps(images),
                tool_call_id=call["id"],
                name=call["name"],
            )
        )
    return {"messages": tool_msgs, "image_urls": images}


def output_node(state: TravelState) -> Dict[str, Any]:
    summary = _maybe_refine_summary(
        state.get("city") or "the city",
        state.get("city_summary"),
        state.get("weather_forecast", []),
        state.get("timeframe", {}),
    )
    payload = {
        "city_summary": summary,
        "weather_forecast": state.get("weather_forecast", []),
        "image_urls": state.get("image_urls", []),
        "city": state.get("city"),
        "source": state.get("source"),
        "timeframe": state.get("timeframe"),
    }
    return {"final_response": payload}


# --------------------------
# Graph factory
# --------------------------


def build_graph() -> Any:
    builder = StateGraph(TravelState)
    builder.add_node("city_info", city_info_node)
    builder.add_node("search_node", search_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("weather_node", weather_node)
    builder.add_node("image_node", image_node)
    builder.add_node("output_node", output_node, join=True)

    builder.add_edge(START, "city_info")
    builder.add_conditional_edges("city_info", route_after_city_info, ["planner_node", "search_node"])
    builder.add_edge("search_node", "planner_node")
    builder.add_edge("planner_node", "weather_node")
    builder.add_edge("planner_node", "image_node")
    builder.add_edge("weather_node", "output_node")
    builder.add_edge("image_node", "output_node")
    builder.add_edge("output_node", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


GRAPH = build_graph()
