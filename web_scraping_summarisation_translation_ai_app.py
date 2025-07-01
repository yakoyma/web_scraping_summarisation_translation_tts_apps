"""
===============================================================================
Project: Web Scraping, Summarisation, and Translation Application with Gradio
===============================================================================
"""
# Standard library
import platform

# Other libraries
import langchain_community
import langchain_core
import spacy
import gradio as gr


from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gradio_pdf import PDF


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('LangChain Community: {}'.format(langchain_community.__version__))
print('LangChain Core: {}'.format(langchain_core.__version__))
print('SpaCy: {}'.format(spacy.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_response(source_language: str, summary_language: str, request: str,
                 url: str, file: str, text: str) -> str:
    """This function summarises a text and translates the summary using an
    AI model.

    Args:
        source_language (str): the language of the text
        summary_language (str): the language of the summary
        request (str): the user's prompt
        url (str): the website url to scrape
        file (str): the path of the PDF file
        text (str): the user's text

    Returns:
        response (str): the response of the model
    """

    try:

        # Check wether the user inputs are valid
        if source_language and summary_language and any([url, file, text]):

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
            elif file: # Check if there is any PDF file path
                # Load the PDF file
                loader = PyPDFLoader(file_path=file)
                document = []
                for page in loader.lazy_load():
                    document.append(page)
                text = ''.join(doc.page_content for doc in document)

            # Instantiate the NLP model
            nlp = spacy.load(name='xx_ent_wiki_sm')
            nlp.add_pipe(factory_name='sentencizer')

            # Cleanse the text
            request = request.strip()
            text = text.strip()

            # Create the prompt template for the user's request
            template = """
            You are a nice and helpful assistant, and your task is to 
            summarise in {summary language} without comments the following 
            text in {source language}:

            Text: {text}

            by giving priority to the user's request, if there is any: 
            {request}."""

            # Check if there is any text and wether the text tokens length is
            # less than the maximum tokens limit selected
            request_tokens_length = len(nlp(request))
            text_tokens_length = len(nlp(text))
            template_tokens_length = len(nlp(template))
            input_tokens_length = (request_tokens_length +
                                   text_tokens_length + template_tokens_length)
            if text_tokens_length > 0 and input_tokens_length < 11000:

                # Selection of the maximum number of tokens at the model input
                max_tokens_length = 2000

                # Iterate over each sentence in the text
                document = nlp(text)
                sentences = [sentence.text for sentence in document.sents]
                summaries_list, sentences_list = [], []
                for i, sentence in enumerate(sentences):
                    sentences_list.append(sentence)
                    current_list_tokens_length = len(nlp(' '.join(
                        sentences_list)))
                    current_input_tokens_length = (request_tokens_length +
                        current_list_tokens_length + template_tokens_length)

                    """
                    Check if adding the current sentence to the sentences list 
                    would exceed the maximum tokens limit:
                    - If yes, stop adding sentences to the list and create 
                      a text. Then, reset the list with the current sentence.
                    - If not, continue to add sentences to the list or create 
                      text if the list is not empty at the last iteration.
                    """
                    if current_input_tokens_length > max_tokens_length:
                        current_text = ' '.join(sentences_list[:-1])
                        sentences_list = [sentence]
                    else:
                        if i < len(sentences) - 1:
                            current_text = ''
                        elif i == len(sentences) - 1 and sentences_list:
                            current_text = ' '.join(sentences_list)

                    # Check if there is any text and the maximum number of
                    # tokens limit at the model input
                    current_text_tokens_length = len(nlp(current_text))
                    if current_text_tokens_length > 0:
                        current_input_tokens_length = (
                            request_tokens_length +
                            current_text_tokens_length + template_tokens_length
                        )

                        # Select the context window size
                        num_ctx = next(pow(2, i + 1) for i in range(7, 14) if
                                       current_input_tokens_length < pow(2, i))

                        # Instantiate the model
                        model = Ollama(
                            model='mistral-nemo:12b-instruct-2407-q8_0',
                            num_ctx=num_ctx,
                            num_predict=-1,
                            temperature=0,
                            top_k=40,
                            top_p=0.8
                        )
                        prompt = PromptTemplate.from_template(template)
                        chain = prompt | model | StrOutputParser()
                        current_response = chain.invoke(
                            {
                                'summary language': summary_language,
                                'source language': source_language,
                                'text': current_text,
                                'request': request
                            }
                        )
                        summaries_list.append(current_response)
                        response = ' '.join(summaries_list)
                    else:
                        response = ('The request is too long or the format of '
                                    'the website, document, or text is '
                                    'unsuitable.')
            else:
                response = ('The text and the request are too long and the '
                            'maximum number of tokens has been exceeded, '
                            'or the text is unreadable.')
        else:
            response = ('Invalid input data. Please complete the fields '
                        'correctly.')

    except Exception as error:
        response = f'The following unexpected error occurred: {error}'
    return response



# Instantiate the app
languages_list = [
    'Arabic', 'Bengali', 'Chinese', 'English', 'French', 'German', 'Hindi',
    'Italian', 'Japanese', 'Korean', 'Malayalam', 'Portuguese', 'Russian',
    'Spanish', 'vietnamese'
]
app = gr.Interface(
    fn=get_response,
    inputs=[
        gr.Dropdown(
            choices=languages_list,
            label='Source language (Supported languages)',
            type='value'
        ),
        gr.Dropdown(
            choices=languages_list,
            label='Summary language (Supported languages)',
            type='value'
        ),
        gr.Textbox(label='Request in supported languages'),
        gr.Textbox(label='url'),
        PDF(label='PDF file'),
        gr.Textbox(label='Text in supported languages')
    ],
    outputs=gr.Textbox(label='Summary'),
    title='Web Scraping, Summarisation, and Translation Application'
)



if __name__ == '__main__':
    app.launch()
