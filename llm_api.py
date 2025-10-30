import openai


class LLMAPI:
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model

    def generate_response(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content
