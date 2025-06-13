"""
===============================================================================
Project: Web Scraping and Translation Application with Gradio
===============================================================================
"""
# Standard libraries
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



def get_response(source_language: str, target_language: str, url: str,
                 file: str, text: str) -> str:
    """This function translates a text using an AI model.

    Args:
        source_language (str): the language of the text
        target_language (str): the language of the translation
        url (str): the website url to scrape
        file (str): the path of the PDF file
        text (str): the user's text

    Returns:
        response (str): the response of the model
    """

    try:

        # Check wether the user inputs are valid
        if (source_language and target_language and
            (source_language != target_language) and any([url, file, text])):

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

            # cleanse the text
            text = text.strip()

            # Create the prompt template
            template = """
            You are a nice and helpful assistant, and your task is to 
            translate the following text from {source language} to 
            {target language}:

            Text: {text}."""

            # Check if there is any text
            text_tokens_length = len(nlp(text))
            template_tokens_length = len(nlp(template))
            input_tokens_length = text_tokens_length + template_tokens_length
            if text_tokens_length > 0 and input_tokens_length < 11000:

                # Selection of the maximum number of tokens at the model input
                max_tokens_length = 2000
                document = nlp(text)
                sentences = [sentence.text for sentence in document.sents]
                translations_list, sentences_list = [], []

                # Iterate over each sentence in the text
                for i, sentence in enumerate(sentences):
                    sentences_list.append(sentence)
                    current_text_tokens_length = len(nlp(' '.join(
                        sentences_list)))

                    """
                    Check if adding the current sentence to the sentences list 
                    would exceed the maximum tokens limit:
                    - If yes, stop adding sentences to the list and create 
                      a text. Then, reset the list with the current sentence.
                    - If not, continue to add sentences to the list or create 
                      text if the list is not empty at the last iteration.
                    """
                    if current_text_tokens_length >= max_tokens_length:
                        current_text = ' '.join(sentences_list[:-1])
                        sentences_list = [sentence]
                    else:
                        if i < len(sentences) - 1:
                            current_text = ''
                        elif i == len(sentences) - 1 and sentences_list:
                            current_text = ' '.join(sentences_list)

                    # Check if there is any text and the maximum number of
                    # tokens limit at the model input
                    template_tokens_length = len(nlp(template))
                    current_text_tokens_length = len(nlp(current_text))
                    current_input_tokens_length = (request_tokens_length +
                        current_text_tokens_length + template_tokens_length)
                    if (current_text_tokens_length > 0 and
                        input_tokens_length < max_tokens_length):

                        # Select the context window size
                        num_ctx = next(pow(2, i + 1) for i in range(7, 14) if
                                       input_tokens_length < pow(2, i))

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
                                'source language': source_language,
                                'target language': target_language,
                                'text': current_text
                            }
                        )
                        translations_list.append(current_response)
                        response = ' '.join(translations_list)
                    else:
                        response = ('The format of the website, document, '
                                    'or text is unsuitable.')
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
    'Arabic', 'Bengali', 'Chinese', 'English', 'French', 'German', 'Hindi',
    'Italian', 'Japanese', 'Korean', 'Malayalam', 'Portuguese', 'Russian',
    'Spanish', 'vietnamese'
]
app = gr.Interface(
    fn=get_response,
    inputs=[
        gr.Dropdown(
            choices=languages_list,
            label='Source language',
            type='value'
        ),
        gr.Dropdown(
            choices=languages_list,
            label='Translation language',
            type='value'
        ),
        gr.Textbox(label='url'),
        PDF(label='PDF file'),
        gr.Textbox(label='Text in supported languages')
    ],
    outputs=gr.Textbox(label='Translation'),
    title='Web Scraping and Translation Application'
)



if __name__ == '__main__':
    app.launch()
