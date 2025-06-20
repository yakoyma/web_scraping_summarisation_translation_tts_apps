"""
===============================================================================
Project: Web Scraping and Summarisation Application with Gradio
===============================================================================
"""
# Standard library
import platform

# Other libraries
import langchain_community
import demoji
import transformers
import spacy
import gradio as gr


from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from demoji import replace
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gradio_pdf import PDF


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('LangChain Community: {}'.format(langchain_community.__version__))
print('Demoji: {}'.format(demoji.__version__))
print('Transformers: {}'.format(transformers.__version__))
print('SpaCy: {}'.format(spacy.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_summary(language: str, url: str, file: str, text: str) -> str:
    """This function summarises a text using a Large Language Model (LLM)
    specialising in summarisation.

    Args:
        language (str): the language of the text
        url (str): the website url to scrape
        file (str): the path of the PDF file
        text (str): the user's text

    Returns:
        response (str): the response of the model
    """

    try:

        # Check wether the user inputs are valid
        if language and any([url, file, text]):

            # Load the dataset
            # Check if there is any url
            if url:
                # Web scraping
                url = url.strip()
                loader = AsyncHtmlLoader(
                    web_path=[url], default_parser='html.parser')
                html_document = loader.load()
                document = BeautifulSoupTransformer().transform_documents(
                    html_document)
                text = ''.join(doc.page_content for doc in document)
            elif file: # Check if there is any PDF file path
                # Load the PDF file
                loader = PyPDFLoader(file_path=file)
                document = []
                for page in loader.lazy_load():
                    document.append(page)
                text = ''.join(doc.page_content for doc in document)

            # Instantiate the summarisation model
            if language == 'English':
                model_name = 'Falconsai/text_summarization'
            elif language == 'French':
                model_name = 'lincoln/mbart-mlsum-automatic-summarization'
            else:
                model_name = 'csebuetnlp/mT5_m2m_crossSum'
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            summariser = pipeline(
                task='summarization', model=model, tokenizer=tokenizer)

            # Cleanse the text
            text = text.strip()
            text = replace(string=text, repl='')

            # Check if there is any text and wether the text tokens length is
            # less than the maximum tokens limit selected
            text_tokens_length = len(tokenizer.tokenize(text))
            if text_tokens_length > 0 and text_tokens_length < 10000:

                # Selection of the maximum number of tokens at the model input
                if text_tokens_length <= 2000:
                    max_tokens_length = 128
                elif text_tokens_length > 2000 and text_tokens_length <= 5000:
                    max_tokens_length = 256
                else:
                    max_tokens_length = 512

                # Instantiate the NLP model
                nlp = spacy.load(name='xx_ent_wiki_sm')
                nlp.add_pipe(factory_name='sentencizer')

                # Iterate over each sentence in the text
                document = nlp(text)
                sentences = [sentence.text for sentence in document.sents]
                summaries_list, sentences_list = [], []
                for i, sentence in enumerate(sentences):
                    sentences_list.append(sentence)
                    current_list_tokens_length = len(tokenizer.tokenize(
                        ' '.join(sentences_list)))

                    """
                    Check if adding the current sentence to the sentences list 
                    would exceed the maximum tokens limit:
                    - If yes, stop adding sentences to the list and create 
                      a text. Then, reset the list with the current sentence.
                    - If not, continue to add sentences to the list or create 
                      text if the list is not empty at the last iteration.
                    """
                    if current_list_tokens_length >= max_tokens_length:
                        current_text = ' '.join(sentences_list[:-1])
                        sentences_list = [sentence]
                    else:
                        if i < len(sentences) - 1:
                            current_text = ''
                        elif i == len(sentences) - 1 and sentences_list:
                            current_text = ' '.join(sentences_list)

                    # Check if there is a text and the maximum number of
                    # tokens limit at the model input
                    current_text_tokens_length = len(tokenizer.tokenize(
                        current_text))
                    if (current_text_tokens_length > 0 and
                        current_text_tokens_length < max_tokens_length):
                        result = summariser(
                            current_text,
                            max_length=current_text_tokens_length,
                            min_length=0,
                            do_sample=False
                        )
                        current_summary = ' '.join(
                            [summ['summary_text'] for summ in result])
                        summaries_list.append(current_summary)
                        response = ' '.join(summaries_list)
                    else:
                        response = ('The format of the website, document, or '
                                    'text is unsuitable.')
            else:
                response = ('The text is too long and the maximum number of '
                            'tokens has been exceeded, or the text is '
                            'unreadable.')
        else:
            response = ('Invalid input data. Please complete the fields '
                        'correctly.')

    except Exception as error:
        response = f'The following unexpected error occurred: {error}'
    return response



# Instantiate the app
languages_list = [
    'Amharic', 'Arabic', 'Azerbaijani', 'Bengali', 'Burmese',
    'Chinese_simplified', 'Chinese_traditional', 'English', 'French',
    'Gujarati', 'Hausa', 'Hindi', 'Igbo', 'Indonesian', 'Japanese', 'Kirundi',
    'Korean', 'Kyrgyz', 'Marathi', 'Nepali', 'Oromo', 'Pashto', 'Persian',
    'Pidgin', 'Portuguese', 'Punjabi', 'Russian', 'Scottish_gaelic',
    'Serbian_cyrillic', 'Serbian_latin', 'Sinhala', 'Somali', 'Spanish',
    'Swahili', 'Tamil', 'Telugu', 'Thai', 'Tigrinya', 'Turkish', 'Ukrainian',
    'Urdu', 'Uzbek', 'Vietnamese', 'Welsh', 'Yoruba'
]
app = gr.Interface(
    fn=get_summary,
    inputs=[
        gr.Dropdown(
            choices=languages_list,
            label='Language of the text (Supported languages)',
            type='value'
        ),
        gr.Textbox(label='url'),
        PDF(label='PDF file'),
        gr.Textbox(label='Text in supported languages')
    ],
    outputs=gr.Textbox(label='Summary'),
    title='Web Scraping and Summarisation Application'
)



if __name__ == '__main__':
    app.launch()
