"""Streamlit demo:  streamlit run app.py"""
import html

import pandas as pd
import streamlit as st

import config
from preprocessing import analyze_text
from spellchecker import HindiSpellChecker

st.set_page_config(page_title="हिंदी वर्तनी जाँचक", page_icon="✍️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if config.VOCAB_FILE.exists():
        return HindiSpellChecker.load(min_freq=1), "data/processed/vocab.json"
    lines = config.SEED_FILE.read_text(encoding="utf8").splitlines()
    return HindiSpellChecker.from_sentences(lines), "seed corpus (demo only - run build_vocab.py)"


sc, source = load_model()

st.title("✍️ हिंदी वर्तनी जाँचक · Hindi Spelling Checker")
st.caption(f"Vocabulary source: {source}")

with st.sidebar:
    st.header("Model")
    for k, v in sc.stats().items():
        st.metric(k.replace("_", " ").title(), f"{v:,}")
    n_sugg = st.slider("Suggestions per word", 1, 8, 4)
    sc.edit_penalty = st.slider("Edit penalty (higher = prefer closer spellings)", 1.0, 8.0, float(sc.edit_penalty), 0.5)
    sc.context_weight = st.slider("Context weight (bigram influence)", 0.0, 3.0, float(sc.context_weight), 0.25)

tab_check, tab_nlp, tab_eval = st.tabs(["Spell check", "NLP pipeline", "Model self-check"])

EXAMPLE = "मुझे हिंदी भाषा बहूत पसद है और हम परीवार के साथ रहते हैं।"

with tab_check:
    text = st.text_area("Enter Hindi text", EXAMPLE, height=150)
    if st.button("Check spelling", type="primary"):
        norm, results = sc.check_text(text, n=n_sugg)
        errors = [r for r in results if not r["known"]]

        out, pos = [], 0
        for r in results:
            out.append(html.escape(norm[pos:r["start"]]))
            if r["known"]:
                out.append(html.escape(r["word"]))
            else:
                tip = ", ".join(r["suggestions"]) or "no suggestion"
                out.append(f'<span title="{html.escape(tip)}" style="background:#ffd6d6;'
                           f'border-bottom:2px solid #d33;padding:0 2px;border-radius:3px">'
                           f'{html.escape(r["word"])}</span>')
            pos = r["end"]
        out.append(html.escape(norm[pos:]))
        st.markdown(f'<div style="font-size:1.3rem;line-height:2.1rem">{"".join(out)}</div>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Words checked", len(results))
        c2.metric("Misspelled", len(errors))
        c3.metric("Accuracy of text", f"{100 * (1 - len(errors) / max(len(results), 1)):.0f}%")

        if errors:
            st.subheader("Suggestions")
            st.dataframe(pd.DataFrame({"Misspelled": [e["word"] for e in errors],
                                       "Suggestions": [" · ".join(e["suggestions"]) or "-" for e in errors]}), hide_index=True)
            st.subheader("Auto-corrected text")
            st.success(sc.correct_text(text))
        else:
            st.success("No spelling mistakes found ✅")

with tab_nlp:
    nlp_text = st.text_area("Text to analyse", "मैं भारत से हूँ और मुझे हिंदी भाषा बहुत पसंद है।", height=100, key="nlp")
    if nlp_text:
        for stage, value in analyze_text(nlp_text).items():
            st.markdown(f"**{stage.replace('_', ' ').title()}**")
            st.code(value if isinstance(value, str) else " | ".join(value), language=None)

with tab_eval:
    st.write("Corrupts real vocabulary words and checks whether the model recovers them.")
    samples = st.slider("Samples", 100, 2000, 300, 100)
    if st.button("Run self-check"):
        import test_model
        with st.spinner("Evaluating..."):
            a1, a3, ms = test_model.test_synthetic(sc, samples, 0.0, 0.0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Top-1 accuracy", f"{a1:.1%}")
        c2.metric("Top-3 accuracy", f"{a3:.1%}")
        c3.metric("Latency", f"{ms:.1f} ms/word")
