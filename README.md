# AI-CHATBOT-WITH-NLP

*COMPANY* : CODTECH IT SOLUTIONS
*NAME* : SAKSHI KAILASH RAMTEKE
*INTERN ID* : CT4MTDF290
*DOMAIN* : PYTHON PROGRAMMING
*DURATION* : 16 WEEKS
*MENTOR* : NEELA SANTHOSH KUMAR

Objectives
- Build a simple AI Chatbot using NLP libraries
- Preprocess and clean user input using spaCy or NLTK
- Retrieve the most relevant answer from a knowledge base
- Provide both CLI and web-based (Streamlit) interaction
- Ensure extensibility to add new questions and answers easily
3. Tools & Technologies
- Python 3.10+
- Libraries: pandas, scikit-learn, spaCy, NLTK, Streamlit
- IDE: Visual Studio Code
- Knowledge base stored as CSV file
  
Step-by-Step Procedure
1. Install Python and VS Code
2. Create and activate a virtual environment
3. Install required libraries using `pip install -r requirements.txt`
4. Download spaCy English model: `python -m spacy download en_core_web_sm`
5. Download NLTK data (stopwords, punkt, wordnet)
6. Run chatbot using `python chatbot.py`
7. Optionally run Streamlit web app using `streamlit run streamlit_app.py`

Code Explanation
The chatbot works as follows:
- Loads Q/A pairs from knowledge_base.csv
- Preprocesses questions using spaCy/NLTK (tokenization, stopword removal, lemmatization)
- Converts questions into vectors (TF-IDF)
- Computes similarity between user query and stored questions
- Returns the best matching answer
If scikit-learn is not available, a token-overlap fallback method is used.

Output
[demo_run.txt](https://github.com/user-attachments/files/22476498/demo_run.txt)

Conclusion
This project demonstrates how a basic AI chatbot can be built using NLP techniques. It highlights the importance of text preprocessing and similarity-based retrieval. The chatbot can be extended by adding more data to the knowledge base, improving retrieval with embeddings, or integrating advanced conversational AI models for generative responses.
