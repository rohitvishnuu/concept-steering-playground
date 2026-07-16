"""
Concept Steering Playground -- interactive Streamlit app.

Run with:
    streamlit run app.py

Drag the slider and watch the same prompt's completion shift along the
chosen concept direction in real time.
"""
import streamlit as st

from steering import DEFAULT_MODEL, SteerableModel, list_concepts

st.set_page_config(page_title="Concept Steering Playground", page_icon="🎛️", layout="centered")

st.title("🎛️ Concept Steering Playground")
st.caption(
    "Drag the slider to add or subtract a concept direction from the model's "
    "activations at generation time -- no fine-tuning, just an inference-time "
    "activation edit (the technique behind demos like Anthropic's Golden Gate Claude, "
    "at a much smaller scale)."
)


@st.cache_resource(show_spinner="Loading model (first run only)...")
def get_model(model_name: str) -> SteerableModel:
    return SteerableModel(model_name)


@st.cache_data(show_spinner="Computing steering vector...")
def get_vector(_model: SteerableModel, model_name: str, concept: str, layer: int):
    # model_name included in cache key since _model itself isn't hashable
    return _model.compute_steering_vector(concept, layer)


model_name = st.sidebar.text_input("Model", value=DEFAULT_MODEL, help="Any small HF causal LM, e.g. gpt2, distilgpt2")
sm = get_model(model_name)

concept = st.sidebar.selectbox("Concept", list_concepts())
layer = st.sidebar.slider("Layer", 0, sm.n_layers - 1, value=min(6, sm.n_layers - 1))
max_new_tokens = st.sidebar.slider("Max new tokens", 10, 100, 40)
seed = st.sidebar.number_input("Seed", value=0, step=1)

vector = get_vector(sm, model_name, concept, layer)

prompt = st.text_area("Prompt", value="I think that the future of technology")
alpha = st.slider(
    f"Steering strength (alpha) -- negative subtracts '{concept}', positive adds it",
    min_value=-10.0, max_value=10.0, value=0.0, step=0.5,
)

if st.button("Generate", type="primary") or "last_result" not in st.session_state:
    with st.spinner("Generating..."):
        text = sm.generate(prompt, max_new_tokens, vector, alpha, seed)
    st.session_state["last_result"] = text

st.markdown("### Output")
st.write(st.session_state.get("last_result", ""))

with st.expander("What's actually happening here?"):
    st.markdown(
        f"""
1. `{len(prompt.split())}`-token prompt is run through **{model_name}**.
2. At layer **{layer}**, we add `alpha * direction` to every token's residual
   stream activation, where `direction` is the mean-activation difference
   between hand-written **{concept}**-positive and **{concept}**-negative
   example sentences (see `data/concept_pairs.json`).
3. Generation proceeds as normal from there -- the model has no idea its
   activations were nudged.
4. `alpha = 0` is the unsteered baseline; try sweeping it from very negative
   to very positive and watch the tone shift.
        """
    )
