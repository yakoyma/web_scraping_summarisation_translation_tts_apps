"""
===============================================================================
Project: Translation Application with Gradio
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


from langchain_community.document_loaders import PyPDFLoader
from demoji import replace
from transformers import M2M100ForConditionalGeneration
from tokenization_small100 import SMALL100Tokenizer
from gradio_pdf import PDF


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('LangChain Community: {}'.format(langchain_community.__version__))
print('Demoji: {}'.format(demoji.__version__))
print('Transformers: {}'.format(transformers.__version__))
print('SpaCy: {}'.format(spacy.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_translation(source_language: int, target_language: int, text: str,
                    file: str) -> str:
    """This function translates a text using a Large Language Model (LLM)
    specialising in translation.

    Args:
        source_language (str): the language of the text
        target_language (str): the language of the translation
        text (str): the user's text
        file (str): the path of the PDF file

    Returns:
        response (str): the translation of the text
    """

    try:

        # Check wether the user inputs are valid
        if (source_language and target_language and (source_language !=
            target_language) and any([text, file])):

            # Load the dataset
            # Check if there is any PDF file path
            if file:
                # Load the PDF file
                loader = PyPDFLoader(file_path=file)
                document = []
                for page in loader.lazy_load():
                    document.append(page)
                text = ''.join(doc.page_content for doc in document)

            # Instantiate the translation model
            model_name = 'alirezamsh/small100'
            model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = SMALL100Tokenizer.from_pretrained(model_name)

            # Cleanse the text
            text = text.strip()
            text = replace(string=text, repl='')

            # Check if there is any text
            text_tokens_length = len(tokenizer.tokenize(text))
            if text_tokens_length > 0 and text_tokens_length < 8000:

                # Selection of languages
                languages_list = [
                    'Afrikaans', 'Albanian', 'Arabic', 'Asturian',
                    'Belarusian', 'Bosnian', 'Bulgarian', 'Catalan; Valencian',
                    'Cebuano', 'Chinese', 'Czech', 'Danish', 'English',
                    'Finnish', 'French', 'Galician', 'German', 'Greek',
                    'Haitian; Haitian Creole', 'Hausa', 'Hebrew', 'Icelandic',
                    'Igbo', 'Iloko', 'Indonesian', 'Italian', 'Javanese',
                    'Korean', 'Latvian', 'Luxembourgish; Letzeburgesch',
                    'Macedonian', 'Malay', 'Norwegian', 'Occitan (post 1500)',
                    'Persian', 'Portuguese', 'Romanian; Moldavian; Moldovan',
                    'Russian', 'Serbian', 'Spanish', 'Sundanese', 'Swahili',
                    'Swedish', 'Tagalog', 'Turkish', 'Ukrainian', 'Urdu',
                    'Vietnamese', 'Western Frisian', 'Xhosa', 'Yoruba'
                ]
                languages_codes_list = [
                    'af', 'sq', 'ar', 'ast', 'be', 'bs', 'bg', 'ca', 'ceb',
                    'zh', 'cz', 'da', 'en', 'fi', 'fr', 'gl', 'de', 'el',
                    'ht', 'ha', 'he', 'is', 'ig', 'ilo', 'id', 'it', 'jv',
                    'ko', 'lv', 'lb', 'mk', 'ms', 'no', 'oc', 'fa', 'pt',
                    'ro', 'ru', 'sr', 'es', 'su', 'sw', 'sv', 'tl', 'tr',
                    'uk', 'ur', 'vi', 'fi', 'xh', 'yo'
                ]
                languages = dict(zip(languages_codes_list, languages_list))
                src_language = next(key for key, value in languages.items() if
                    source_language == value)
                tgt_language = next(key for key, value in languages.items() if
                    target_language == value)

                # Selection of the maximum number of tokens at the model input
                max_tokens_length = 256

                # Instantiate the NLP model
                nlp = spacy.load(name='xx_ent_wiki_sm')
                nlp.add_pipe(factory_name='sentencizer')

                # Iterate over each sentence in the text
                document = nlp(text)
                sentences = [sentence.text for sentence in document.sents]
                translations_list, sentences_list = [], []
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
                        tokenizer.tgt_lang = tgt_language
                        encoded = tokenizer(current_text, return_tensors='pt')
                        generated_tokens = model.generate(**encoded)
                        result = tokenizer.batch_decode(
                            generated_tokens, skip_special_tokens=True)
                        current_translation = ' '.join(result)
                        translations_list.append(current_translation)
                        response = ' '.join(translations_list)
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
    'Afrikaans', 'Albanian', 'Arabic', 'Asturian', 'Belarusian', 'Bosnian',
    'Bulgarian', 'Catalan; Valencian', 'Cebuano', 'Chinese', 'Czech',
    'Danish', 'English', 'Finnish', 'French', 'Galician', 'German', 'Greek',
    'Haitian; Haitian Creole', 'Hausa', 'Hebrew', 'Icelandic', 'Igbo',
    'Iloko', 'Indonesian', 'Italian', 'Javanese', 'Korean', 'Latvian',
    'Luxembourgish; Letzeburgesch', 'Macedonian', 'Malay', 'Norwegian',
    'Occitan (post 1500)', 'Persian', 'Portuguese',
    'Romanian; Moldavian; Moldovan', 'Russian', 'Serbian', 'Spanish',
    'Sundanese', 'Swahili', 'Swedish', 'Tagalog', 'Turkish', 'Ukrainian',
    'Urdu', 'Vietnamese', 'Western Frisian', 'Xhosa', 'Yoruba'
]
app = gr.Interface(
    fn=get_translation,
    inputs=[
        gr.Dropdown(
            choices=languages_list, label='Source language', type='value'),
        gr.Dropdown(
            choices=languages_list, label='Target language', type='value'),
        gr.Textbox(label='Text in supported languages'),
        PDF(label='PDF file'),
    ],
    outputs=gr.Textbox(label='Translation'),
    title='Translation Application'
)



if __name__ == '__main__':
    app.launch()
