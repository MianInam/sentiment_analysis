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
    """Calculate sentiment score focusing on adjectives and log their scores."""
    pos_score = 0.0
    neg_score = 0.0
    adj_count = 0

    tokens = word_tokenize(comment)
    tagged_tokens = pos_tag(tokens)

    print("DEBUG: Adjectives and Scores")  # Log Header for Debugging

    for word, tag in tagged_tokens:
        # Only consider adjectives
        if tag.startswith('J'):
            lemma = lemmatizer.lemmatize(word, pos=wn.ADJ)
            synsets = wn.synsets(lemma, pos=wn.ADJ)

            # Exclude invalid words like "i"
            if word.lower() in {"i"}:
                continue
            if synsets:
                synset = synsets[0]
                try:
                    swn_synset = swn.senti_synset(synset.name())
                    adj_pos = swn_synset.pos_score()
                    adj_neg = swn_synset.neg_score()
                    pos_score += adj_pos
                    neg_score += adj_neg
                    adj_count += 1

                    # Log each adjective with its positive and negative scores
                    print(f"Adjective: {word}, Pos: {adj_pos}, Neg: {adj_neg}")

                except Exception as e:
                    # Handle exceptions for missing SentiWordNet entries
                    print(f"Adjective: {word}, Error: {e}")
                    continue

    # If no adjectives are found, return neutral
    if adj_count == 0:
        return 'neutral'

    # Calculate sentiment based on adjective scores
    sentiment_score = pos_score - neg_score
    print(f"DEBUG: Final Scores -> Pos: {pos_score}, Neg: {neg_score}, Sentiment Score: {sentiment_score}")  # Log final scores
    if sentiment_score > 0:  
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'


@app.route('/analyze_comments', methods=['POST'])
def analyze_comments():
    data = request.json
    comments = data.get('comments', [])

    if not isinstance(comments, list):
        return jsonify({'error': 'Invalid input format, expected a list of comments'}), 400

    positive_count = 0
    neutral_count = 0
    negative_count = 0
    comment_results = []

    for comment_data in comments:
        comment_text = comment_data.get('comment', '').lower()
        if not comment_text:
            continue

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

    if not isinstance(posts, list):
        return jsonify({'error': 'Invalid input format, expected a list of posts'}), 400

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
            comment_text = comment_data.get('comment', '').lower()
            if not comment_text:
                continue

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
            'id': post.get('id'),
            'title': post.get('title'),
            'content': post.get('content'),
            'created_at': post.get('created_at'),
            'updated_at': post.get('updated_at'),
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
