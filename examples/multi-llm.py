from asymetry import init_observability
import openai
import anthropic

# Single initialization - both work!
init_observability(service_name="my-app")

# OpenAI - tracked automatically
openai_client = openai.OpenAI()
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Anthropic - also tracked automatically
anthropic_client = anthropic.Anthropic()
response = anthropic_client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)

import time
time.sleep(6)

print("Done!")