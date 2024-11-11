from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)
lemmatizer = WordNetLemmatizer()

def nltk_pos_tagger(nltk_tag):
    """Convert PennTreebank tags to WordNet tags."""
    if nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('R'):
        return wn.ADV
    return None

def enhance_with_lemmas(comment):
    """Lemmatize the input comment using WordNet."""
    tokens = word_tokenize(comment)
    tagged_tokens = pos_tag(tokens)
    enriched_comment = []

    for word, tag in tagged_tokens:
        wn_tag = nltk_pos_tagger(tag)
        if wn_tag:
            enriched_comment.append(lemmatizer.lemmatize(word, pos=wn_tag))
        else:
            enriched_comment.append(word)

    return " ".join(enriched_comment)

def get_sentiment_score(comment):
    """Calculate the sentiment score for the input comment with improved negation and context handling."""
    pos_score = 0.0
    neg_score = 0.0
    tokens_count = 0
    tokens = word_tokenize(comment)
    tagged_tokens = pos_tag(tokens)
    
    negation = False  # Track if the last word was a negation
    
    for word, tag in tagged_tokens:
        wn_tag = nltk_pos_tagger(tag)
        
        # Handle negation
        if word.lower() in ["not", "no", "never", "none", "hardly", "barely",'dont',]:
            negation = True
            continue
        
        if wn_tag:
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            synsets = wn.synsets(lemma, pos=wn_tag)
            if synsets:
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
                if swn_synset:
                    # Weight for adjectives and adverbs
                    weight = 1.0
                    if wn_tag in {wn.ADJ, wn.ADV}:
                        weight = 1.5  # Give more weight to adjectives and adverbs
                    pos_score += swn_synset.pos_score() * weight
                    neg_score += swn_synset.neg_score() * weight
                    tokens_count += 1

                    # Reverse scores if negation was detected
                    if negation:
                        pos_score, neg_score = neg_score, pos_score
                        negation = False  # Reset negation after applying

    # Calculate sentiment based on updated thresholds
    if tokens_count == 0:
        return 'neutral'

    sentiment_score = pos_score - neg_score
    if sentiment_score > 0.3:  # Adjust thresholds based on your testing
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

@app.route('/analyze_comments', methods=['POST'])
def analyze_comments():
    data = request.json
    comments = data.get('comments', [])

    positive_count = 0
    neutral_count = 0
    negative_count = 0
    comment_results = []

    for comment_data in comments:
        comment_text = comment_data['comment'].lower()
        enriched_comment = enhance_with_lemmas(comment_text)
        sentiment = get_sentiment_score(enriched_comment)

        if sentiment == 'positive':
            positive_count += 1
        elif sentiment == 'negative':
            negative_count += 1
        else:
            neutral_count += 1

        comment_result = comment_data.copy()
        comment_result['status'] = sentiment
        comment_results.append(comment_result)

    response = {
        'data': {
            'total': {
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count
            },
            'comments': comment_results,
        }
    }

    return jsonify(response)

@app.route('/analyze_posts', methods=['POST'])
def analyze_posts():
    data = request.json
    posts = data.get('posts', [])

    overall_positive = 0
    overall_neutral = 0
    overall_negative = 0
    post_results = []

    for post in posts:
        post_positive = 0
        post_neutral = 0
        post_negative = 0
        comments_with_sentiment = []

        for comment_data in post.get('comments', []):
            comment_text = comment_data['comment'].lower()
            enriched_comment = enhance_with_lemmas(comment_text)
            sentiment = get_sentiment_score(enriched_comment)

            if sentiment == 'positive':
                post_positive += 1
            elif sentiment == 'negative':
                post_negative += 1
            else:
                post_neutral += 1

            comment_result = comment_data.copy()
            comment_result['status'] = sentiment
            comments_with_sentiment.append(comment_result)

        # Update overall sentiment counts
        overall_positive += post_positive
        overall_neutral += post_neutral
        overall_negative += post_negative

        # Collect the results for each post
        post_results.append({
            'id': post['id'],
            'title': post['title'],
            'content': post['content'],
            'created_at': post['created_at'],
            'updated_at': post['updated_at'],
            'total': {
                'positive': post_positive,
                'neutral': post_neutral,
                'negative': post_negative
            },
            'comments': comments_with_sentiment
        })

    # Final response structure with overall sentiment totals and post-specific totals
    response = {
        'data': {
            'overall_total': {
                'positive': overall_positive,
                'neutral': overall_neutral,
                'negative': overall_negative
            },
            'posts': post_results,
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
