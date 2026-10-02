import streamlit as st
from src.agent import run_weather_agent


st.set_page_config(page_title="Weather Safety Advisor", page_icon="🌦️", layout="centered")

st.title("🌦️ Weather Safety Advisor")

st.markdown("""
**Heading outside?** 
Before you step out, let's check what the skies have in store. Whether you're planning to cycle, jog, or just take a walk—ask me to make sure the conditions are safe! 🚴‍♂️🌬️
""")

st.caption("Assignment by Shakshyam Pandey • shakshyampandey23@gmail.com")
st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "session_005"


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("E.g., 'Can I cycle in Bhopal today?' or 'Is it safe to go for a run?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking current conditions and safety policies..."):
            response = run_weather_agent(prompt, st.session_state.session_id)
            
           
            raw_ans = response["answer"]
            if isinstance(raw_ans, list):
                final_text = raw_ans[0].get("text", str(raw_ans))
            else:
                final_text = str(raw_ans)

         
            citations = response.get("citations")
            if citations:
                final_text += f"\n\n*Policy Citations: {', '.join(citations)}*"
                
    
            st.markdown(final_text)

  
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_text
    })