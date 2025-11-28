import streamlit as st
import asyncio


from multi_agent import chat_with_agent


st.set_page_config(
    page_title="Mechanical Parts AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Mechanical Parts AI Assistant")
st.caption("Enter a part number (e.g., C-AN00) or a question about parts.")


user_input = st.text_input("Your query:", placeholder="Enter your query here...")


if st.button("Ask Agent"):
    
    if user_input:
        
        with st.spinner("Agent is thinking..."):
            try:
                
                response = asyncio.run(chat_with_agent(user_input))

                
                st.subheader("Agent's Response:")
                st.markdown(response)

            except Exception as e:
                
                st.error(f"An error occurred: {e}")
    else:
        
        st.warning("Please enter a query before asking the agent.")