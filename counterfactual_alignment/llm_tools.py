
from google.genai import types

def generate_review(client,review1,review2,think=False):

    preprompt = "We are trying to generate a spectrum of sentiments. Here are two film reviews. Generate a review that is sentimentally between the two reviews by modifying words which control the sentiment. Edit as little as possible outside of these words. Do not explain your response, provide only the review text. Here are the reviews:\n\n"
    text = preprompt + "Review 1: " + review1 + "\n\nReview 2: " + review2

    if think:
        config = {}
    else:
        config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    )

        response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=text,
    config=config,
    )

        return response  