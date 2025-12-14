import asyncio
import uuid
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from travel_agent import GRAPH


load_dotenv(override=False)


st.set_page_config(page_title="Multi-Modal Travel Assistant", page_icon="🫀", layout="wide")
st.title("Multi-Modal Travel Assistant")
st.caption("LangGraph + manual tool execution + mock APIs (weather, images, search)")


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state["history"] = []


def render_response(data: Dict[str, Any]) -> None:
    summary = data.get("city_summary")
    city = data.get("city")
    timeframe = data.get("timeframe") or {}
    source = data.get("source") or "mock"
    st.write(f"**City:** {city or 'Unknown'} | **Source:** {source} | **Window:** {timeframe.get('label', 'this week')}")
    if summary:
        st.markdown(summary)
    else:
        st.warning("No city summary available yet.")

    col1, col2 = st.columns([1, 1])
    with col1:
        forecast: List[Dict[str, Any]] = data.get("weather_forecast") or []
        if forecast:
            df = pd.DataFrame(forecast)
            if "date" in df and "temperature_c" in df:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                st.line_chart(df["temperature_c"], height=240)
            else:
                st.write(forecast)
        else:
            st.info("Weather data not ready.")

    with col2:
        images = data.get("image_urls") or []
        if images:
            st.image(images, width=220, caption=[city for _ in images])
        else:
            st.info("Images not ready.")


user_prompt = st.chat_input("Ask about a city (e.g., 'Tell me about Kyoto' or 'What about next week?')")
if user_prompt:
    with st.spinner("Let me gather details..."):
        result = asyncio.run(
            GRAPH.ainvoke(
                {"messages": [HumanMessage(content=user_prompt)]},
                config={"configurable": {"thread_id": st.session_state["thread_id"]}},
            )
        )
    response = result.get("final_response") or {}
    st.session_state["history"].append({"user": user_prompt, "assistant": response})


for turn in st.session_state["history"]:
    st.chat_message("user").write(turn["user"])
    with st.chat_message("assistant"):
        render_response(turn["assistant"])



