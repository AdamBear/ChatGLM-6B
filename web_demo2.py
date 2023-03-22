from transformers import AutoModel, AutoTokenizer
import streamlit as st
from streamlit_chat import message


st.set_page_config(
    page_title="ChatGLM-6b 演示",
    page_icon=":robot:"
)


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("/data/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("/data/chatglm-6b", trust_remote_code=True).half().cuda()
    model = model.eval()
    return tokenizer, model


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2



def predict(input, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []

    if len(history) > 1:
        for i, (query, response) in enumerate(history[:-1]):
            message(query, avatar_style="big-smile", key=str(i) + "_user")
            message(response, avatar_style="bottts", key=str(i))

    for response, history in model.stream_chat(tokenizer, input, history):
        query, response = history[-1]
        key = str(len(history)-1)
        if key in st.session_state:
            del st.session_state[key]
        message(query, avatar_style="big-smile", key=key + "_user")
        message(response, avatar_style="bottts", key=key)

    return history


# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")


placeholder = st.empty()

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, st.session_state["state"])

    st.balloons()