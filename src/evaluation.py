from summac.model_summac import SummaCConv
import streamlit as st

@st.cache_data(show_spinner=False, ttl=600)
def summac_consistency_detection(model_specs, documents, llm_response) -> dict:
    model = SummaCConv(**model_specs)
    documents = documents
    llm_response = llm_response
    score = model.score([documents], [llm_response])
    
    return score

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Evaluation:
        summac={"models":["vitc"],
                "bins":'percentile',
                'granularity':'sentence',
                'nli_labels':'e',
                'device':'cuda',
                'start_file':'.models/summac_conv_vitc_sent_perc_e.bin',
                'agg':'mean'}

    score = summac_consistency_detection(Evaluation.summac, "Quebec is in Canada", "Canada is in Quebec")
    print(score)