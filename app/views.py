import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def ask_question(query, context=[]):
    # combines the new question with a previous context
    context += [query]

    # given the most recent context (4096 characters)
    # continue the text up to 2048 tokens ~ 8192 charaters
    completion = openai.Completion.create(
        engine='text-davinci-003',  # one of the most capable models available
        prompt='\n\n'.join(context)[:4096],
        max_tokens=2048,
        temperature=0.0,  # Lower values make the response more deterministic
    )

    # append response to context
    response = completion.choices[0].text.strip('\n')
    context += [response]

    # list of (user, bot) responses. We will use this format later
    responses = [(u, b) for u, b in zip(context[::2], context[1::2])]

    return responses, context


def evaluate_score(responses):
    # Scoring system based on the Patient Health Questionnaire (PHQ-9)
    score = 0

    for response in responses:
        if 'not at all' in response.lower():
            score += 0
        elif 'several days' in response.lower():
            score += 1
        elif 'more than half the days' in response.lower():
            score += 2
        elif 'nearly every day' in response.lower():
            score += 3

    return score


def anxiety_filter(score: int) -> str:
    # Interpret the score
    status = "Not Anxiety"
    if score >= 10:
        status = "Severe Anxiety"
    elif score >= 5:
        status = "Mild Anxiety"
    else:
        status = "Not Anxiety"
    return status


def follow_up_key(key):
    # Define the key for the follow-up question
    return key + '_follow_up'


def assess_mental_health(answers):
    # Combine all answers into a single string
    text = " ".join(answers)

    # Use sentiment analysis to assess the overall mood of the patient's responses
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)['compound']
    # if sentiment_scores['compound'] < -0.5:
    #     print("The patient's responses suggest a negative overall mood.")
    # else:
    #     print("The patient's responses suggest a neutral or positive overall mood.")

    # Check for keywords related to depression
    depression_keywords = ['sad', 'hopeless', 'empty', 'worthless', 'guilty', 'suicidal']
    depression_count = sum(text.count(keyword) for keyword in depression_keywords)
    # if depression_count >= 2:
    #     print("The patient's responses suggest symptoms of depression.")
    # else:
    #     print("The patient's responses do not suggest symptoms of depression.")

    return sentiment_scores, depression_count


def depression_filter(depression_score: float) -> str:
    status = "Not Depressed"
    if depression_score <= 0.5:
        status = "Not Depressed"
    elif 0.5 < depression_score <= 0.6:
        status = "Mild Depressed"
    elif 0.6 < depression_score <= 0.8:
        status = "Moderate Depressed"

    elif depression_score > 0.8:
        status = "Severe Depressed"

    else:
        status = "Not Depressed"
    return status


def suicidal_filter(suicidal_score: float) -> str:
    status = "Not Suicidal"
    if suicidal_score <= 0.5:
        status = "Not Suicidal"
    elif 0.5 < suicidal_score <= 0.6:
        status = "Mild Suicidal"
    elif 0.6 < suicidal_score <= 0.8:
        status = "Moderate Suicidal"

    elif suicidal_score > 0.8:
        status = "Severe Suicidal"

    else:
        status = "Not Suicidal"
    return status
