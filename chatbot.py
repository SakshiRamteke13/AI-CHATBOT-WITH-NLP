# chatbot.py - Simple retrieval-based chatbot using TF-IDF (scikit-learn) + spaCy/NLTK preprocessing.
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from preprocess import Preprocessor

KB_CSV = 'knowledge_base.csv'

def build_index(kb_path=KB_CSV):
    df = pd.read_csv(kb_path)
    pre = Preprocessor()
    df['processed'] = df['question'].astype(str).apply(lambda t: pre.join(t))
    # Try to use scikit-learn's TF-IDF; if not available, fall back to token-overlap scoring
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['processed'].tolist())
        return {'mode':'tfidf','df':df,'vectorizer':vectorizer,'X':X,'pre':pre,'cosine_similarity':cosine_similarity}
    except Exception as e:
        # fallback: store token sets
        df['tokens'] = df['processed'].apply(lambda s: set(s.split()))
        return {'mode':'overlap','df':df,'pre':pre}

def answer_query(state, query, top_k=1, threshold=0.2):
    pre = state['pre']
    q_processed = pre.join(query)
    if state['mode'] == 'tfidf':
        vec = state['vectorizer'].transform([q_processed])
        sims = state['cosine_similarity'](vec, state['X'])[0]
        idxs = np.argsort(-sims)[:top_k]
        results = []
        for i in idxs:
            results.append((float(sims[i]), state['df'].iloc[i]['answer'], state['df'].iloc[i]['question']))
        if results and results[0][0] >= threshold:
            return results
        else:
            return results  # return best matches even if score low
    else:
        q_tokens = set(q_processed.split())
        scores = []
        for r in state['df'].itertuples():
            overlap = q_tokens.intersection(r.tokens)
            score = len(overlap) / max(len(q_tokens),1)
            scores.append(score)
        idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idxs:
            results.append((float(scores[i]), state['df'].iloc[i]['answer'], state['df'].iloc[i]['question']))
        return results

def chat_loop():
    state = build_index()
    print('AI Chatbot ready. Type your question (type "exit" to quit).')
    while True:
        q = input('\nYou: ').strip()
        if q.lower() in ('exit','quit'):
            print('Goodbye!')
            break
        matches = answer_query(state, q, top_k=3)
        if not matches:
            print('Bot: Sorry, I do not have an answer for that yet.')
            continue
        # print top match + confidence and show alternatives
        top_score, top_answer, top_question = matches[0]
        print(f'Bot (score={top_score:.3f}): {top_answer}')
        if len(matches) > 1:
            print('\nOther close matches:')
            for s, ans, ques in matches[1:]:
                print(f' - (score={s:.3f}) Q: {ques} -> A: {ans}')

if __name__ == '__main__':
    # ensure we run from project folder
    Path(KB_CSV).absolute()
    chat_loop()
