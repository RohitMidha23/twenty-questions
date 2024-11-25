from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


HOST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a host of a game show playing the common game 20 qestions. You have chosen the topic '{topic}'. The guesser will ask you a question and you need to answer whether the question is about the topic or not. Respond with Yes if it is about the topic, No otherwise.",
        ),
        ("human", "Question: {question}"),
    ]
)

### Guesser Prompts
RECOMMENDER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strategic recommender in a 20 questions game. Your goal is to:
            1. Decide whether to make a guess or continue questioning.
            2. If guessing, come up with a list of top candidates.
            2. If guessing, provide confidence scores for top candidates

            Previous conversation history and answers will help refine the candidate list.
            Only recommend guessing if confidence is very high (>90%) for a specific candidate.
            Previous candidates: {candidates}
        """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "human",
            "Based on the conversation history, should we guess or continue questioning? What are the current possible candidates?",
        ),
    ]
)

QUESTION_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at creating binary search style questions for 20 questions game.
            Your goal is to create questions that will eliminate approximately half of the remaining candidates.
            
            Current candidates: {candidates}
            
            Rules:
            1. Questions must be answerable with Yes/No
            2. Each question should target a property that divides the candidate pool
            3. Avoid questions that only eliminate 1-2 candidates
            4. Consider previous questions to avoid repetition
        """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "human",
            "Generate a question that will effectively split the candidate pool. {feedback}",
        ),
    ]
)

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a question evaluator for 20 questions game. Assess if the proposed question is effective by checking:
            1. Will it eliminate roughly half the candidates?
            2. Is it clear and unambiguous?
            3. Does it avoid overlap with previous questions?
            4. Is it a yes/no question?
            
            Current candidates: {candidates}
            Proposed question: {question}
            Expected elimination: {expected_elimination}
            Expected retention: {expected_retention}
        """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "human",
            "Evaluate if this is a good binary search question. If not, suggest improvements.",
        ),
    ]
)
