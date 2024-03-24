"""
===============================================================================
Web Scraping using Selenium of LangChain library and Automatic Text
Summarisation with Transformers library
===============================================================================

This file is organised as follows:
1. Web Scraping
2. Text Preprocessing
3. Text Summarisation
4. Text-to-Speech
"""
# Standard libraries
import platform

# Other libraries
import langchain_community
import nltk
import demoji
import contractions
import processtext as pt
import spacy
import bertopic
import langchain
import transformers
import gtts

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import SpacyTextSplitter
from nltk import sent_tokenize, word_tokenize
from demoji import findall, replace
from contractions import fix
from processtext import clean_l
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gtts import gTTS

print('\nPython: {}'.format(platform.python_version()))
print('LangChain community: {}'.format(langchain_community.__version__))
print('NLTK: {}'.format(nltk.__version__))
print('Demoji: {}'.format(demoji.__version__))
print('Processtext: {}'.format(pt.__version__))
print('SpaCy: {}'.format(spacy.__version__))
print('BERTopic: {}'.format(bertopic.__version__))
print('LangChain: {}'.format(langchain.__version__))
print('Transformers: {}'.format(transformers.__version__))
print('gTTS: {}'.format(gtts.__version__))



# 1. Web Scraping
url = ('https://academy.finxter.com/university/prompt-engineering-with'
       '-python-and-openai')

try:
    loaders = SeleniumURLLoader(urls=[url])
    data = loaders.load()
    for result in data:
        texts = result.page_content



    # 2. Text Preprocessing
    # Cleanse the text
    print(f'\n\nNumber of sentences: {len(sent_tokenize(texts))}')
    print(f'Number of words: {len(word_tokenize(texts))}')

    # Selection of relevant information
    texts = ' '.join(sent_tokenize(texts)[1:66])

    # Remove emojis if needed
    print(f'\nA mapping of emojis of the text:\n{findall(string=texts)}')
    texts = replace(string=texts, repl='')

    # Fix contractions if needed
    texts = fix(texts)

    # Restore punctuations if needed
    texts = texts.replace(' ,', ',')
    texts = texts.replace(' .', '.')
    texts = texts.replace(' ?', '?')
    texts = texts.replace(' !', '!')
    texts = texts.replace(' )', ')')
    texts = texts.replace('( ', '(')
    texts = texts.replace(') ,', '),')
    texts = texts.replace(') .', ').')
    texts = texts.replace(') ?', ')?')
    texts = texts.replace(') !', '!.')
    texts = texts.replace('" ,', '",')
    texts = texts.replace('" .', '".')
    texts = texts.replace('" ?', '"?')
    texts = texts.replace('" !', '"!')

    # Remove extra spaces if needed and selection of relevant information
    sentences = []
    for sentence in sent_tokenize(texts):
        sentence = sentence.replace('\u200d', '')
        sentence = clean_l(
            text=sentence, clean_all=False, extra_spaces=True)
        sentences.append(' '.join(sentence).strip())
    print(f'\nNumber of sentences after cleansing and selection:'
          f' {len(sentences)}')
    print(f'Sentences after cleansing and selection:\n{sentences}')

    # Join sentences
    texts = ' '.join(sentences)
    print(f'\nTexts after cleansing and selection:\n{texts}')

    # Topic modeling
    nlp = spacy.load(name='en_core_web_sm')
    doc = nlp(texts)
    model = BERTopic(embedding_model=spacy.load(
        name='en_core_web_md',
        exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']))
    topics, probs = model.fit_transform([sent.text for sent in doc.sents])
    print(model.get_topic_info())
    document = model.get_document_info([sent.text for sent in doc.sents])
    print(document[document.Representative_document==True])
    print(list(document[document.Representative_document==True].Document))

    # Split the text into chuncks
    text_splitter = SpacyTextSplitter(separator=' ', chunk_size=1500)
    prompts = text_splitter.split_text(texts)
    print(f'\n\nPrompts:\n{prompts}')
    print(f'\nTotal number of prompts: {len(prompts)}')
    for i in range(len(prompts)):
        print(f'\nNumber of sentences: {len(sent_tokenize(prompts[i]))}')
        print(f'Number of words: {len(word_tokenize(prompts[i]))}')
        print(f'Prompt {i + 1}:\n{prompts[i]}')



    # 3. Text Summarisation
    # Flax Community t5 Base CNN dm model
    flax_t5_base_cnn_tokenizer = AutoTokenizer.from_pretrained(
        'flax-community/t5-base-cnn-dm')
    flax_t5_base_cnn_model = AutoModelForSeq2SeqLM.from_pretrained(
        'flax-community/t5-base-cnn-dm')
    flax_t5_base_cnn_summarizer = pipeline(
        task='summarization',
        model=flax_t5_base_cnn_model,
        tokenizer=flax_t5_base_cnn_tokenizer)

    # Text summaries
    flax_t5_base_cnn_result = flax_t5_base_cnn_summarizer(
        prompts, max_length=150, min_length=50, do_sample=False)
    print(f'\n\nFlax Community t5 Base CNN model '
          f'result:\n{flax_t5_base_cnn_result}')

    # Join summaries
    flax_t5_base_cnn_text = ' '.join(
        [summ['summary_text'] for summ in flax_t5_base_cnn_result])
    print(f'\nFlax Community t5 Base CNN '
          f'text summary:\n{flax_t5_base_cnn_text}')

    # Save as text file
    with open('flax_t5_base_cnn_text_summary.txt', 'w') as f:
        f.write(flax_t5_base_cnn_text)


    # Facebook Bart Large CNN model
    bart_tokenizer = AutoTokenizer.from_pretrained(
        'facebook/bart-large-cnn')
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(
        'facebook/bart-large-cnn')
    bart_summarizer = pipeline(
        task='summarization',
        model=bart_model,
        tokenizer=bart_tokenizer)

    # Text summaries
    bart_result = bart_summarizer(
        prompts, max_length=150, min_length=50, do_sample=False)
    print(f'\n\nFacebook Bart model result:\n{bart_result}')

    # Join summaries
    bart_text = ' '.join([summ['summary_text'] for summ in bart_result])
    print(f'\nFacebook Bart model text summary:\n{bart_text}')

    # Save as text file
    with open('bart_text_summary.txt', 'w') as f:
        f.write(bart_text)


    # Google Pegasus CNN Dailymail model
    pegasus_cnn_dailymail_tokenizer = AutoTokenizer.from_pretrained(
        'google/pegasus-cnn_dailymail')
    pegasus_cnn_dailymail_model = AutoModelForSeq2SeqLM.from_pretrained(
        'google/pegasus-cnn_dailymail')
    pegasus_cnn_dailymail_summarizer = pipeline(
        task='summarization',
        model=pegasus_cnn_dailymail_model,
        tokenizer=pegasus_cnn_dailymail_tokenizer)

    # Text summaries
    pegasus_cnn_dailymail_result = pegasus_cnn_dailymail_summarizer(
        prompts, max_length=150, min_length=50, do_sample=False)
    print(f'\n\nGoogle Pegasus CNN Dailymail model '
          f'result:\n{pegasus_cnn_dailymail_result}')

    # Join summaries
    pegasus_cnn_dailymail_text = ' '.join(
        [summ['summary_text'] for summ in pegasus_cnn_dailymail_result])
    print(f'\nGoogle Pegasus CNN Dailymail model '
          f'text summary:\n{pegasus_cnn_dailymail_text}')

    # Save as text file
    with open('pegasus_cnn_dailymail_text_summary.txt', 'w') as f:
        f.write(pegasus_cnn_dailymail_text)


    # Falconsai Text Summarization model
    falconsai_tokenizer = AutoTokenizer.from_pretrained(
        'Falconsai/text_summarization')
    falconsai_model = AutoModelForSeq2SeqLM.from_pretrained(
        'Falconsai/text_summarization')
    falconsai_summarizer = pipeline(
        task='summarization',
        model=falconsai_model,
        tokenizer=falconsai_tokenizer)

    # Text summaries
    falconsai_result = falconsai_summarizer(
        prompts, max_length=150, min_length=50, do_sample=False)
    print(f'\n\nFalconsai model result:\n{falconsai_result}')

    # Join summaries
    falconsai_text = ' '.join(
        [summ['summary_text'] for summ in falconsai_result])
    print(f'\nFalconsai model text summary:\n{falconsai_text}')

    # Save as text file
    with open('falconsai_text_summary.txt', 'w') as f:
        f.write(falconsai_text)



    # 4. Text-to-Speech
    # Convert texts to audiobooks using gTTS (Google Text-to-Speech)
    gTTS(text=flax_t5_base_cnn_text, lang='en').save(
        savefile='flax_t5_base_cnn_text_summary_audiobook.wav')
    gTTS(text=bart_text, lang='en').save(
        savefile='bart_text_summary_audiobook.wav')
    gTTS(text=pegasus_cnn_dailymail_text, lang='en').save(
        savefile='pegasus_cnn_dailymail_text_summary_audiobook.wav')
    gTTS(text=falconsai_text, lang='en').save(
        savefile='falconsai_text_summary_audiobook.wav')
except Exception as error:
    print(f'\n\nThe following unexpected error occurred: {error}')
