"""
===============================================================================
Project: Web Scraping, Short Summarisation, and Text-to-Speech Application
with Gradio
===============================================================================
"""
# Standard library
import platform

# Other libraries
import langchain_community
import langchain_core
import transformers
import gtts
import gradio as gr


from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer
from gtts import gTTS
from gradio_pdf import PDF


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('LangChain Community: {}'.format(langchain_community.__version__))
print('LangChain Core: {}'.format(langchain_core.__version__))
print('Transformers: {}'.format(transformers.__version__))
print('gTTS: {}'.format(gtts.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_response(language: str, request: str, url: str, file: str,
                 text: str) -> str:
    """This function summarises a text briefly using an AI model and convert
    the summary into speech using a Text-to-Speech model.

    Args:
        language (str): the language of the text
        request (str): the user's prompt
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
            if url:
                # Web scraping
                url = url.strip()
                loader = AsyncHtmlLoader(
                    web_path=[url], default_parser='html.parser')
                html_document = loader.load()
                document = BeautifulSoupTransformer().transform_documents(
                    html_document)
                text = ''.join(doc.page_content for doc in document)
            elif file: # Check if there is any PDF file
                # Load the PDF file
                loader = PyPDFLoader(file_path=file)
                document = []
                for page in loader.lazy_load():
                    document.append(page)
                text = ''.join(doc.page_content for doc in document)

            # Instantiate the tokenizer
            model_name = 'microsoft/Phi-3.5-mini-instruct'
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)

            # Cleanse the text and the user's request
            request = request.strip()
            text = text.strip()

            # Create the prompt template for the user's request
            template = """
            You are a nice and helpful assistant, and your task is to summarise 
            the following text concisely without exceeding 100 tokens:

            Text: {text} 

            by respecting the user's request, if there is any: {request}."""

            # Check if there is any text and whether the number of tokens at
            # the model input is less than the maximum value selected
            request_tokens_length = len(tokenizer.tokenize(request))
            text_tokens_length = len(tokenizer.tokenize(text))
            template_tokens_length = len(tokenizer.tokenize(template))
            input_tokens_length = (request_tokens_length +
                                   text_tokens_length + template_tokens_length)
            if text_tokens_length > 0 and input_tokens_length < 16000:

                # Select the context window size
                num_ctx = next(pow(2, i + 1) for i in range(7, 15) if
                    input_tokens_length < pow(2, i))

                # Instantiate the model
                model = Ollama(
                    model='phi3.5:3.8b-mini-instruct-q8_0',
                    num_ctx=num_ctx,
                    num_predict=100,
                    temperature=0,
                    top_k=40,
                    top_p=0.8
                )
                prompt = PromptTemplate.from_template(template)
                chain = prompt | model | StrOutputParser()
                response = chain.invoke({'text': text, 'request': request})

                # Select the gTTS language code
                languages_list = [
                    'Chinese (Mandarin)',
                    'English',
                    'French',
                    'Portuguese',
                    'Spanish'
                ]
                languages_codes_list = ['zh-CN', 'en', 'fr', 'pt', 'es']
                languages = dict(zip(languages_codes_list, languages_list))
                gtts_language = next(
                    key for key, value in languages.items() if
                    language == value
                )

                # Instantiate the Text-to-Speech model
                tts = gTTS(text=response, lang=gtts_language)
                audio = 'response.wav'
                tts.save(audio)
            else:
                response = ('The text and the request are too long and the '
                            'maximum number of tokens has been exceeded, '
                            'or the text is unreadable.')
                audio = None
        else:
            response = ('Invalid input data. Please complete the fields '
                        'correctly.')
            audio = None

    except Exception as error:
        response = f'The following unexpected error occurred: {error}'
        audio = None
    return response, audio



# Instantiate the app
languages_list = [
    'Chinese (Mandarin)', 'English', 'French', 'Portuguese', 'Spanish']
app = gr.Interface(
    fn=get_response,
    inputs=[
        gr.Dropdown(
            choices=languages_list, label='Source language', type='value'),
        gr.Textbox(label='Request in supported languages'),
        gr.Textbox(label='url'),
        PDF(label='PDF file'),
        gr.Textbox(label='Text in supported languages')
    ],
    outputs=[
        gr.Textbox(label='Summary'), gr.Audio(label='Summary audio')],
    title='Web Scraping, Short Summarisation, and Text-to-Speech Application'
)



if __name__ == '__main__':
    app.launch()
