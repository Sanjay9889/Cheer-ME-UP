import os

import openai
from fastapi import FastAPI

# Configure OpenAI API credentials
from starlette.middleware.cors import CORSMiddleware

from app.depression import Depression
from app.questions import questions, subjective_question
from app.schema import CheerMeUpCreate
from app.views import (
    evaluate_score,
    depression_filter,
    suicidal_filter,
    anxiety_filter, ask_question)

# Set up API key and authorization
openai.api_key = os.environ["OPENAI_API_KEY"]

app = FastAPI()

def add_middleware(app):
    origins = [
        "http://localhost",
        "http://localhost:8080",
        "*"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["DELETE", "GET", "POST", "PUT"],
        allow_headers=["*"],
    )

# Define route to generate GPT-3 response based on user input
@app.post("/cheer-me-up/")
async def generate_text(payload: CheerMeUpCreate) -> dict:
    try:
        # Ask the questions and collect the responses
        responses = []
        context = f"I am a {payload.age}-year-old {payload.gender}."
        data = payload.dict()
        print(f"Data - {data}")
        one_word_question = data["one_word_question"]
        for key, answers in one_word_question.items():
            # response = ask_question(answers, context=context)
            responses.append(answers)

            # # Ask a follow-up question if needed
            # if 'could you elaborate' in response.lower():
            #     follow_up_question = questions.get(follow_up_key(key))
            #     response = ask_question(follow_up_question, context=context)
            #     responses.append(response)

        print(f"One word response - {responses}")
        subjective_question = data["subjective_question"]
        print(f"Subjective Question {subjective_question}")
        subjective_answers = [value for key, value in subjective_question.items()]
        # mha = MentalHealthAnalyzer()
        # sentiment_response = mha.assess_mental_health(subjective_answers)
        # print(f"Sentiment Response {sentiment_response}")

        depression = Depression()
        scores = depression.predict(subjective_answers)
        depression_score = scores[0]
        suicidal_score = scores[1]

        # Evaluate the responses and provide a recommendation
        score = evaluate_score(responses)

        return {
            "status": 200,
            'depression': depression_filter(depression_score),
            "suicidal": suicidal_filter(suicidal_score),
            "anxiety": anxiety_filter(score)}
    except Exception as e:
        return {
            "status": 400,
            "message": f"Unable to complete the request and error was {e}"
        }


@app.get("/list-questions/")
async def list_questions():
    return {
        "status": 200,
        'questions': questions,
        'subjective_questions': subjective_question
    }


# Define API endpoint for chatbot
@app.post("/chat")
async def chatbot(msg: str):
    responses, context = ask_question(msg, context=[])

    return {
        "user": responses[-1][0],
        "bot": responses[-1][1]
    }
