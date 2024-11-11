import nltk

# Define the download directory

# Download the required NLTK packages to 'C:\\nltk_data'
nltk.download('punkt', download_dir='C:\\nltk_data')
nltk.download('sentiwordnet', download_dir='C:\\nltk_data')
nltk.download('punkt_tab', download_dir='C:\\nltk_data')
nltk.download('wordnet', download_dir='C:\\nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='C:\\nltk_data')
nltk.download('averaged_perceptron_tagger_eng', download_dir='C:\\nltk_data')


nltk.data.path.append('C:\\nltk_data')