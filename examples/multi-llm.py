from asymetry import init_observability, observe
import openai

# Single initialization - both work!
init_observability(service_name="my-app")

# OpenAI - tracked automatically
openai_client = openai.OpenAI()


@observe(span_type="workflow")
def fun():
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi, give me a 50 worded summary on pilots!"}],
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Give me 50 words about good food?"}]
    )


fun()

import time

time.sleep(6)

print("Done!")
