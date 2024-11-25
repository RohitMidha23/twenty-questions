from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

INITIAL_HOST_PROMPT_v1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a host of a game show playing the common game 20 qestions. You need to choose an object or living thing to start the game with. This will be the topic of the game.",
        ),
        ("human", "Choose a topic:"),
    ]
)
HOST_PROMPT_v1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a host of a game show playing the common game 20 qestions. You have chosen the topic '{topic}'. The guesser will ask you a question and you need to answer with a yes or no. Yes if the question is about the topic, No otherwise.",
        ),
        ("human", "Question: {question}"),
    ]
)

GUESSER_PROMPT_v1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a guesser in a game show playing the common game 20 qestions. You need to ask the host a question and get a yes or no answer. Your question should be a yes/no question that will help you guess the specific object or living thing that the host has chosen. You have to come up with a question that has not been asked before. You have {question_count} questions left.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "What is your next question?"),
    ]
)

GUESSER_PROMPT_v2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a guesser in a game show playing the common game 20 qestions. You need to ask the host a question and get a yes or no answer. Your question should be a yes/no question that will help you guess the specific object or living thing that the host has chosen. You have to come up with a question that has not been asked before. Start with broader questions to ascertain the topic. As number of questions decrease, your question should become more specific and start to guess the specific object or living thing that the host has chosen. You have {question_count} questions left.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "What is your next question?"),
    ]
)

HOST_PROMPT_v2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a host of a game show playing the common game 20 qestions. You have chosen the topic '{topic}'. The guesser will ask you a question and you need to answer with a yes or no. Yes if the question is about the topic, No otherwise. You also need to determine if the guess is correct or not. \
            Examples: \
            Topic: dog Question: Is it a living thing? \nResponse: Yes Correct_Guess: False \
            Topic: dog Question: Is it an animal? \nResponse: Yes Correct_Guess: False \
            Topic: car Question: Is it a plant? \nResponse: No Correct_Guess: False \
            Topic: car Question: Is it a car? \nResponse: Yes Correct_Guess: True \
            ",
        ),
        ("human", "Question: {question}"),
    ]
)

HOST_PROMPT_v3 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a host of a game show playing the common game 20 qestions. You have chosen the topic '{topic}'. The guesser will ask you a question and you need to answer with a yes or no. Yes if the question is about the topic, No otherwise. You also need to determine if the guess is correct or not. \
            Examples: \
            Topic: dog Question: Is it a living thing? \nResponse: Yes Correct_Guess: False Analysis: A dog is a living thing but the guesser has not yet guessed that it is a dog. \
            Topic: dog Question: Is it an animal? \nResponse: Yes Correct_Guess: False Analysis: A dog is an animal but the guesser has not yet guessed that it is a dog. \
            Topic: car Question: Is it a plant? \nResponse: No Correct_Guess: False Analysis: A car is not a plant. \
            Topic: car Question: Is it a car? \nResponse: Yes Correct_Guess: True Analysis: The guesser has guessed the topic, car, correctly. \
            ",
        ),
        ("human", "Question: {question}"),
    ]
)

GUESSER_PROMPT_v3 = GUESSER_PROMPT_v2


def get_default_prompts():

    return INITIAL_HOST_PROMPT_v1, HOST_PROMPT_v1, GUESSER_PROMPT_v1
