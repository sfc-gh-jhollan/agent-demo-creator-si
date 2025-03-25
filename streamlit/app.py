import uuid
import streamlit as st
import sys
import os
from langgraph.types import Command


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../agent")))

from agent import app as agent


# Initialize Streamlit app
st.title("Demo Creator Chat Interface")

# Session state to manage context
if "messages" not in st.session_state:
    st.session_state.messages = []
if "status_updates" not in st.session_state:
    st.session_state.status_updates = []
if "thread_config" not in st.session_state:
    st.session_state.thread_config = None


def langgraph_stream(prompt):
    if "interrupt" in st.session_state and st.session_state.interrupt:
        thread_config = st.session_state.thread_config  # Reuse stored thread_config
        inputs = Command(resume=prompt)  # Use Command for resuming
        st.session_state.interrupt = None  # Clear the interrupt after resuming
    else:
        thread_config = {
            "configurable": {"thread_id": uuid.uuid4()}
        }  # New thread_config
        st.session_state.thread_config = (
            thread_config  # Store thread_config in session_state
        )
        inputs = {"question": prompt}  # Use plain inputs for new prompts

    for stream_mode, *chunk in agent.app.stream(
        inputs, stream_mode=["messages", "custom"], config=thread_config
    ):
        message_chunk = chunk[0]

        if stream_mode == "custom":
            st.session_state.status_updates.append(message_chunk)
            chain_of_thought.update(
                label="\n\n".join(
                    [f"- {message}" for message in st.session_state.status_updates]
                )
            )
        elif (
            stream_mode == "messages"
            and isinstance(message_chunk[1], dict)
            and "langgraph_node" in message_chunk[1]
            and message_chunk[1]["langgraph_node"].startswith("Display")
        ):
            yield message_chunk[0].content.replace("\n", "\n\n").replace("$", "\\$")

    state = agent.app.get_state(thread_config)

    # Check for interrupts in tasks and store in session state
    for task in state.tasks:
        if task.interrupts:
            st.session_state.interrupt = task.interrupts[0].value
            break


# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chain_of_thought = st.status("Thinking...")
        stream = langgraph_stream(prompt)
        try:
            response = st.write_stream(stream)
        except Exception as e:
            print("Error:", e)
            response = ""

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Reset status updates
    chain_of_thought.update(
        label="\n\n".join(
            [f"- {message}" for message in st.session_state.status_updates]
        ),
        expanded=False,
        state="complete",
    )
    st.session_state.status_updates = []
