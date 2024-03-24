"""
===============================================================================
Web Scraping using Beautifoul Soup Transformer of LangChain library and
Automatic Text Summarisation with Transformers library
===============================================================================

This file is organised as follows:
1. Web Scraping
2. Text Preprocessing
3. Text Summarisation
4. Text-to-Speech
"""
# Standard libraries
import platform
import re

# Other libraries
import langchain_community
import nltk
import demoji
import processtext as pt
import spacy
import bertopic
import langchain
import transformers
import gtts

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import SpacyTextSplitter
from nltk import sent_tokenize, word_tokenize
from demoji import findall, replace
from processtext import clean_l
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gtts import gTTS

print('\nPython: {}'.format(platform.python_version()))
print('Re: {}'.format(re.__version__))
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
url = 'https://www.data-bird.co/blog/web-scraping-python'

try:
    loaders = AsyncHtmlLoader(web_path=[url], default_parser='html.parser')
    html_document = loaders.load()
    document = BeautifulSoupTransformer().transform_documents(
        html_document, tags_to_extract=['h1', 'h2', 'h4', 'p'])
    for result in document:
        texts = result.page_content



    # 2. Text Preprocessing
    # Cleanse the text
    print(f'\n\nNumber of sentences: {len(sent_tokenize(texts))}')
    print(f'Number of words: {len(word_tokenize(texts))}')

    # Remove emojis if needed
    print(f'\nA mapping of emojis of the text:\n{findall(string=texts)}')
    texts = replace(string=texts, repl='')

    # Restore punctuations if needed
    texts = re.sub(
        pattern=r'\s*\(\s*(https?:\/\/\S+)\s*\)\s*', repl='', string=texts)
    texts = texts.replace('( ', '(')
    texts = texts.replace('’ ', '’')
    texts = texts.replace('“ ', '“')
    texts = texts.replace(' ,', ',')
    texts = texts.replace(' .', '.')
    texts = texts.replace(' )', ')')
    texts = texts.replace(' ”', '”')
    texts = texts.replace(';', '; ')
    texts = texts.replace(':', ': ')
    texts = texts.replace(') ,', '),')
    texts = texts.replace(') .', ').')
    texts = texts.replace('. )', '.)')
    texts = texts.replace('? )', '?)')
    texts = texts.replace('! )', '!)')
    texts = texts.replace('” ,', '”,')
    texts = texts.replace('” .', '”.')
    texts = texts.replace('" ,', '",')
    texts = texts.replace('" .', '".')

    # Remove extra spaces if needed
    sentences = []
    for sentence in sent_tokenize(texts):
        sentence = sentence.replace('\u200d', '')
        sentence = clean_l(
            text=sentence, clean_all=False, extra_spaces=True)
        sentences.append(' '.join(sentence).strip())
    print(f'\nSentences after cleansing:\n{sentences}')

    # Join sentences
    texts = ' '.join(sentences)
    print(f'\nTexts after cleansing:\n{texts}')

    # Topic modeling
    nlp = spacy.load(name='fr_core_news_sm')
    doc = nlp(texts)
    model = BERTopic(language='multilingual', embedding_model=spacy.load(
        name='fr_core_news_md',
        exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']))
    topics, probs = model.fit_transform([sent.text for sent in doc.sents])
    print(model.get_topic_info())
    document = model.get_document_info([sent.text for sent in doc.sents])
    print(document[document.Representative_document==True])
    print(list(document[document.Representative_document==True].Document))

    # Split the text into chuncks
    text_splitter = SpacyTextSplitter(separator=' ', chunk_size=2300)
    prompts = text_splitter.split_text(texts)
    print(f'\n\nPrompts:\n{prompts}')
    print(f'\nTotal number of prompts: {len(prompts)}')
    for i in range(len(prompts)):
        print(f'\nNumber of sentences: {len(sent_tokenize(prompts[i]))}')
        print(f'Number of words: {len(word_tokenize(prompts[i]))}')
        print(f'Prompt {i + 1}:\n{prompts[i]}')



    # 3. Text Summarisation
    # Lincoln MBart Mlsum Automatic Summarization model
    tokenizer = AutoTokenizer.from_pretrained(
        'lincoln/mbart-mlsum-automatic-summarization')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        'lincoln/mbart-mlsum-automatic-summarization')
    summarizer = pipeline(
        task='summarization', model=model, tokenizer=tokenizer)

    # Text summaries
    result = summarizer(
        prompts, max_length=150, min_length=50, do_sample=False)
    print(f'\n\nLincoln MBart model result:\n{result}')

    # Join summaries
    text = ' '.join([summ['summary_text'] for summ in result])
    print(f'\nLincoln Mbart model text summary:\n{text}')

    # Save as text file
    with open('lincoln_mbart_text_summary.txt', 'w') as f:
        f.write(text)



    # 4. Text-to-Speech
    # Convert texts to audiobooks using gTTS (Google Text-to-Speech)
    gTTS(text=text, lang='fr').save(
        savefile='lincoln_mbart_text_summary_audiobook.wav')
except Exception as error:
    print(f'\n\nThe following unexpected error occurred: {error}')
