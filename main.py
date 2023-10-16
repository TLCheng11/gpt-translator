import langchain_helper as lch
import streamlit as st


st.title("GPT-translator demo")

target_language = st.text_input(
    label="Enter the language you want to translate into:",
    max_chars=20,
    value="English",
)

source_text = st.text_area(
    label="Enter source text here:",
    max_chars=100,
    placeholder="This is for demo purpose, max input charaters is 100...",
)

clicked = st.button(label="Translate")

if clicked:
    if source_text:
        placeholder = st.text("Translating...")
        response, cb = lch.translate_text(target_language, source_text).values()
        st.write(response)
        placeholder.empty()
        st.divider()
        st.caption(f"Request cost data:")
        st.caption(f"input_char_count: {len(source_text)}")
        st.caption(f"input_token_count: {cb.prompt_tokens}")
        st.caption(f"output_char_count: {len(response)}")
        st.caption(f"output_token_count: {cb.completion_tokens}")
        st.caption(f"total_cost: ${cb.total_cost}")
    else:
        st.write("Input cannot be empty!")

if __name__ == "__main__":
    pass
