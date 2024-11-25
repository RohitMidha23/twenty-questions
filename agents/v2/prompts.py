from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

HOST_PROMPT_v1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a host of a game show playing the common game 20 qestions. You have chosen the topic '{topic}'. The guesser will ask you a question and you need to answer whether the question is about the topic or not. Respond with Yes if it is about the topic, No otherwise.",
        ),
        ("human", "Question: {question}"),
    ]
)

GUESSER_RECOMMENDER_PROMPT_v1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert guesser in a game show playing the common game 20 qestions. \
            The host has chosen a topic which is a specific object or living thing. \
            You can choose to either guess the topic or ask a question. \
            Your job is to come up with a list of possible guesses and questions that can be asked to the host. \
            Some examples of topics include: apple, car, dog, etc. \
            ",
        ),
        # Messages placeholder inserts the entire conversation history
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Come up with a list of possible guesses and questions."),
    ]
)
GUESSER_EVALUATOR_PROMPT_v1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert evaluator who is helping the guesser in a game show playing the common game 20 qestions. \
            The guesser has come up with a list of possible guesses and questions about the topic which is a specific object or thing. \
            Your job is to evaluate each of them and come up with one final guess or question that the guesser should use to guess the topic. \
            Here is the list of guesses and questions: \
            <guesses>{guesses}</guesses> \
            <questions>{questions}</questions> \
            Your analysis should be based on the following criteria: \
            1. At this point in the game, should I be asking a question or making a guess? \
            2. Do I have enough information to make a guess? \
            3. Is the question going to help me come closer to guessing the topic or object? \
            4. Is the guess a specific object or thing? \
            5. Is the guess going to help me eliminate other potential guesses or similarities? \
            Based on these criteria, come up with either just a guess or a question. You have {question_count} questions left. \
            ",
        ),
        # Messages placeholder inserts the entire conversation history
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)

GUESSER_EVALUATOR_PROMPT_v2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert evaluator who is helping the guesser in a game show playing the common game 20 qestions. \
            The guesser has come up with a list of possible guesses and questions about the topic which is a specific object or thing. \
            Your job is to evaluate each of them and come up with one final guess or question that the guesser should use to guess the topic. \
            Here is the list of guesses and questions: \
            <guesses>{guesses}</guesses> \
            <questions>{questions}</questions> \
            Your analysis should be based on the following criteria: \
            1. At this point in the game, should I be asking a question or making a guess? \
            2. Do I have enough information to make a guess? \
            3. Is the question going to help me come closer to guessing the topic or object? \
            4. Is the guess a specific object or thing? \
            5. Is the guess going to help me eliminate other potential guesses or similarities? \
            Based on these criteria, come up with either just a guess or a question. You have {question_count} questions left. \
            At no point should you repeat a question that has already been asked. Do not guess the same thing twice either. When a guess is wrong, reevaluate and try to gather more information before guessing again. \
            ",
        ),
        # Messages placeholder inserts the entire conversation history
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)
