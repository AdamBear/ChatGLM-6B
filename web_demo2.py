from transformers import AutoModel, AutoTokenizer
import streamlit as st
from streamlit_chat import message

# tokenizer = AutoTokenizer.from_pretrained("/data/chatglm-6b", trust_remote_code=True)
# #model = AutoModel.from_pretrained("/data/chatglm-6b", trust_remote_code=True).half().cuda()
# model = AutoModel.from_pretrained("/data/chatglm-6b", trust_remote_code=True).half().quantize(4).cuda()

st.set_page_config(
    page_title="ChatGLM-6b 演示",
    page_icon=":robot:"
)

@st.cache_resource
def get_model():
    # tokenizer = AutoTokenizer.from_pretrained("d:/apps/nlp/models/chatglm-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("d:/apps/nlp/models/chatglm-6b", trust_remote_code=True).half().cuda()
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
    response, history = model.chat(tokenizer, input, history)

    updates = []
    for query, response in history:
        updates.append("用户：" + query)
        message(input, avatar_style="avataaars")
        updates.append("ChatGLM-6B：" + response)
        message(response, avatar_style="big-smile")

    if len(updates) < MAX_BOXES:
        updates = updates + [""] * (MAX_BOXES - len(updates))

    return updates


# create a prompt text for the text generation
prompt_text = st.text_area(label="用户输入",
            height = 200,
            placeholder="请在这儿输入您的命令",
            value="作为抖音本地生活服务的短视频运营专家，我每次会给你键入命令，命令是负责输入商品类型，当你收到我输入商品类型时，你负责帮我每次写出10个吸引人的短视频标题，每个标题限制在40字内，当我键入“继续”时你需要再写出10个")

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, st.session_state["state"])
        st.success("已成功给出回答")

    st.balloons()

st.session_state