import argparse
import logging

import streamlit as st

from config import BaseChatbotInterfaceConfig, ChatbotInterfaceConfig, vLLMChatbotInterfaceConfig, vLLMRAGConfig, QuotationsConfig, OllamaRAGConfig
from context import _retrieve_tokenizer
from profiling import compile_profiled_stats
from rag_utils.db_config import VectorDBDataFiles
from tools import _load_vector_database, retrieve_database_stream
from translations import Translator

logging.basicConfig(filename='.debugging/debugging.log', level=logging.DEBUG)

class App:
    def __init__(self, 
                 engine="ollama", 
                 eval_mode=False, 
                 is_remote=False,
                 hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig):
        self.eval_mode = eval_mode
        self.is_remote = is_remote
        self.engine = engine
        self.hyperparams = hyperparams

        self.config: BaseChatbotInterfaceConfig = ChatbotInterfaceConfig if self.engine=="ollama" else vLLMChatbotInterfaceConfig

        self.model = self.config.default_model_local if not self.is_remote else self.config.default_model_remote
        self.nb_previous_questions = self.config.nb_previous_questions
        self.db_name = self.config.db_name

        self.quotations_mode_status = QuotationsConfig.direct_quotations_mode
        user_language = "fr" if st.context.locale and st.context.locale.startswith('fr') else "en"

        # Initialize language in session state if not present
        if 'language' not in st.session_state:
            st.session_state.language = user_language

        self.system_prompt = ""
        self.translator = Translator(st.session_state.language)

        if "expander_state" not in st.session_state:
            st.session_state["expander_state"] = True

        if "messages" in st.session_state and len(st.session_state.messages) > 0:
            last_message_is_user = st.session_state.messages[-1]["role"] == "user"
            if last_message_is_user:
                st.session_state.messages.pop()
            
    def close_expander(self):
        st.session_state["expander_state"] = False

    def open_expander(self):
        st.session_state["expander_state"] = True
    
    def sidebar_config(self):
        with st.sidebar:

            st.markdown(f"# {self.translator.get('sidebar.language')}")
            
            # Create two columns for language buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("EN", use_container_width=True, type="primary" if st.session_state.language == "en" else "secondary"):
                    st.session_state.language = "en"
                    st.rerun()
            with col2:
                if st.button("FR", use_container_width=True, type="primary" if st.session_state.language == "fr" else "secondary"):
                    st.session_state.language = "fr"
                    st.rerun()

            st.markdown(f"# {self.translator.get('sidebar.title')}")

            self.is_remote = st.toggle(label=self.translator.get('sidebar.remote_mode'), 
                                       value=self.is_remote,
                                       help=self.translator.get('sidebar.remote_mode_tooltip'))

            self.quotations_mode_status = st.toggle(label=self.translator.get('sidebar.direct_quotations'), 
                                                    value=QuotationsConfig.direct_quotations_mode,
                                                    key="quotations_toggle")
            
            model_shortlist = self.config.models_shortlist_local if not self.is_remote else self.config.models_shortlist_remote
            default_model = self.config.default_model_local if not self.is_remote else self.config.default_model_remote
            
            self.model = st.selectbox(label=self.translator.get('sidebar.model_prompt'), 
                                options=model_shortlist, 
                                index=model_shortlist.index(default_model))
            
            orig_db_list = list(VectorDBDataFiles.databases.keys())
            
            translated_db_list = [self.translator.get(f'databases.{db_name}') for db_name in orig_db_list]
            translated_db_name = self.translator.get(f'databases.{self.db_name}')
            
            selected_db_name = st.selectbox(label=self.translator.get('sidebar.db_name'), 
                                options=list(translated_db_list), 
                                index=translated_db_list.index(translated_db_name))
            
            self.db_name = orig_db_list[translated_db_list.index(selected_db_name)]
            
            sources_number = st.number_input(self.translator.get('sidebar.sources_prompt'), 
                                    value=5, 
                                    placeholder="Type a number...",
                                    help=self.translator.get('sidebar.sources_prompt_tooltip'))
            
            self.nb_previous_questions = st.number_input(label=self.translator.get('sidebar.previous_questions'),
                                    value=self.nb_previous_questions,
                                    min_value=0,
                                    help=self.translator.get('sidebar.previous_questions_tooltip'))
            
            with st.popover(self.translator.get('sidebar.advanced_settings'), use_container_width=True, icon=":material/component_exchange:"):
                self.system_prompt = st.text_area(label=str((self.translator.get('sidebar.system_prompt'))), 
                                              height=300,
                                              key="custom_prompt",
                                              value="",
                                              placeholder=self.translator.get('sidebar.system_prompt_placeholder'))
            
            if st.button(self.translator.get('sidebar.reset_button'), 
                         icon=":material/refresh:",
                         key='new_chat', 
                         help=self.translator.get('sidebar.reset_help'),
                         use_container_width=True,
                         on_click=self.open_expander):  
                st.session_state.messages = []
                st.rerun()
                

        return self.model, sources_number, self.system_prompt
    
    def main(self):

        def start_application():
            with st.expander(self.translator.get('about.title'), 
                             expanded=st.session_state["expander_state"],
                             icon=":material/info:"):
                
                st.markdown(f'''
                {self.translator.get('about.welcome')}\n\n
                * {self.translator.get('about.documents_tab')}
                * {self.translator.get('about.metadata_tab')}\n
                {self.translator.get('about.note')}\n\n
                {self.translator.get('about.TOS')}
                ''')

            _load_vector_database(st.session_state.language, self.db_name)
            _retrieve_tokenizer()

        def create_tabs(nb_previous_questions, all_previous_messages):
            tabs_titles = [
                self.translator.get('tabs.response'),
                self.translator.get('tabs.documents'),
                self.translator.get('tabs.metadata')
            ]

            all_previous_assistant_messages = [message for message in all_previous_messages if message["role"] == "assistant"]

            if nb_previous_questions == 0 or len(all_previous_assistant_messages) == 0:
                return st.tabs(tabs_titles)
            
            tabs_titles.append(self.translator.get('tabs.previous_documents'))
            tabs_titles.append(self.translator.get('tabs.previous_metadata'))

            previous_documents = []
            previous_metadata = []

            previous_assistant_messages = all_previous_assistant_messages[-nb_previous_questions:]
            for message in previous_assistant_messages[::-1]:
                previous_documents.append(message["docs"]) # docs = text
                previous_metadata += message["metadata"]

            tabs = st.tabs(tabs_titles)

            joined_previous_documents = "\n\n\n".join(previous_documents)
            display_tabs_docs_and_metadata(tabs[3], tabs[4], previous_metadata, joined_previous_documents)

            return tabs[0], tabs[1], tabs[2]
        
        def display_tabs_docs_and_metadata(tab_docs, tab_metadata, metadata, docs):
            with tab_docs:
                with st.container(border=True, height=500):
                    st.markdown(docs)

            with tab_metadata:
                with st.container(border=True, height=500):
                    text_sources = "\n\n".join(str(*[(str(k)+": "+str(v)) for k,v in i.items()]) for i in metadata)
                    st.markdown(f"""Sources:\n\n{text_sources}""")

        def display_download_button(original_answer, key):
            st.download_button(
                label=self.translator.get('download'), 
                data=original_answer, 
                file_name="chat_conversation.txt",
                key=key,
                on_click='ignore'
            )

        def display_retrieval_messages(eval_mode=self.eval_mode):
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                tab1, tab2, tab3 = create_tabs(self.nb_previous_questions, st.session_state.messages[:-1])

                with st.spinner(self.translator.get('processing'), show_time=False):
                    result, metadata, docs, chunks = retrieve_database_stream(
                        st.session_state.messages[-1]["content"],
                        language=st.session_state.language,
                        db_name=self.db_name,
                        is_remote=self.is_remote,
                        hyperparams=self.hyperparams,
                        n_results=sources_number,
                        chat_model=self.model,
                        quotations_mode=self.quotations_mode_status,
                        custom_system_prompt=self.system_prompt,
                        previous_messages=st.session_state.messages[:-1],
                        nb_previous_questions=self.nb_previous_questions,
                        engine=self.engine
                    )
                    
                    # Populate the tabs with the retrieved documents and metadata as soon as they are ready
                    display_tabs_docs_and_metadata(tab2, tab3, metadata, docs)

                    answer = original_answer = ""

                    with st.container():
                        with tab1:
                            # Create two containers for the response
                            previous_container = st.empty()
                            current_container = st.empty()

                            # Process remaining states
                            for previous_paragraphs, original_previous_paragraphs, stream_generator in result:
                                # Update both containers
                                previous_container.markdown(previous_paragraphs, unsafe_allow_html=True)
                                if stream_generator is not None:
                                    current_container.write_stream(stream_generator)
                                else:
                                    current_container.markdown("")

                                answer = previous_paragraphs
                                original_answer = original_previous_paragraphs

                    st.session_state.messages.append({"role": "assistant", "content":answer, "metadata":metadata, "docs":docs, "chunks":chunks, "original_answer":original_answer, "nb_previous_questions":self.nb_previous_questions})
                    #st.session_state.previous_question_chunks = chunks
                    st.session_state.profiling_counter+=1
  
                    if eval_mode:
                        from evaluation import summac_consistency_detection
                        from config import Evaluation

                        score = summac_consistency_detection(model_specs=Evaluation.summac,
                                                             documents=docs,
                                                             llm_response=original_answer)
                        
                        with open(".evaluation/scores.txt", 'a', encoding='utf-8') as f:
                            f.writelines(f"\n{str(score)}")

                display_download_button(original_answer, key=f"download_{st.session_state.profiling_counter}")

        def process_incoming_input():
            if user_input := st.chat_input(self.translator.get('chat_input'),
                                           on_submit=self.close_expander):
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role":"user", "content":user_input})
                st.empty()

        st.set_page_config(
            page_title=self.translator.get('title'),
            initial_sidebar_state="auto",
            layout="centered",
            menu_items={"Report a bug": "mailto:pierreolivier.bonin@hrsdc-rhdcc.gc.ca",
                        "About":"Developed by Pierre-Olivier Bonin, Ph.D."}
        )

        # Load CSS file
        with open('styles/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        st.title(self.translator.get('title'))

        _, sources_number, self.system_prompt = self.sidebar_config()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "previous_question_chunks" not in st.session_state:
            st.session_state.previous_question_chunks = []

        start_application()

        previous_messages = []

        for idx, message in enumerate(st.session_state.messages):
            is_assistant = message["role"] == "assistant"
            if is_assistant:
                tab1, tab2, tab3 = create_tabs(message["nb_previous_questions"], previous_messages)
                with tab1:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)

                display_tabs_docs_and_metadata(tab2, tab3, message.get("metadata", ""), message.get("docs", ""))
                display_download_button(message.get("original_answer", ""), key=f"download_{idx//2}") # only 1 in 2 messages (the assistant's) should have a download button
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

            previous_messages.append(message)

        process_incoming_input()

        display_retrieval_messages()

class ProfilingApp(App):
    
    def sidebar_config(self):
        # call super() to get the base App's sidebar config
        model, sources_number, system_prompt = super().sidebar_config()

        with st.sidebar:            
            if st.button('Compile profiling', 
                        help='Compile all question-profiling statistics'):
                compile_profiled_stats()
        return model, sources_number, system_prompt


if __name__ == '__main__':
    default_mode = ChatbotInterfaceConfig.default_mode
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("mode", # as per [this issue](https://discuss.streamlit.io/t/command-line-arguments/386/3) keyword args don't work
                        help="Choose the mode in which you want to run this app",
                        choices=["local", "remote", "evaluation_local", "evaluation_remote", "profiling_local", "profiling_remote", "vllm"],
                        type=str,
                        nargs='?',
                        default=default_mode)

    args = parser.parse_args()
    print(args)
    
    if not hasattr(st.session_state, 'profiling_counter'):
        st.session_state.profiling_counter = 0
    
    is_vllm = args.mode in ["vllm"]
    is_remote = args.mode in ["remote", "evaluation_remote", "profiling_remote"]
    is_evaluation = args.mode in ["evaluation_local", "evaluation_remote"]
    is_profiling = args.mode in ["profiling_local", "profiling_remote"]
    location_mode_name = "Remote" if is_remote else "Local"
    hyperparams = vLLMRAGConfig.HyperparametersAccuracyConfig if is_vllm else OllamaRAGConfig.HyperparametersAccuracyConfig
    engine = "vllm" if is_vllm else "ollama"


    if is_profiling:

        import cProfile, io, pstats  # noqa: E401

        def prof_to_csv(prof: cProfile.Profile, sortby):
            out_stream = io.StringIO()
            pstats.Stats(prof, stream=out_stream).sort_stats(sortby).print_stats()
            result = out_stream.getvalue()
            result = 'ncalls' + result.split('ncalls')[-1]
            lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
            return '\n'.join(lines)
            
        with cProfile.Profile() as pr:
            profiling_app = ProfilingApp(is_remote=is_remote, engine=engine, hyperparams=hyperparams)
            profiling_app.main()
            print(f"This is question #{st.session_state.profiling_counter}")
            s = io.StringIO
            sortby = pstats.SortKey.TIME
            if st.session_state.profiling_counter>0:
                csv = prof_to_csv(pr, sortby=sortby)
                with open(f".profiling/prof_stats_Q{st.session_state.profiling_counter}.csv", "w") as f:
                    f.write(csv)

    elif is_evaluation:
        evaluation_app = App(eval_mode=True, is_remote=is_remote)
        evaluation_app.main()

    else:
        app = App(is_remote=is_remote, engine=engine, hyperparams=hyperparams)
        app.main()