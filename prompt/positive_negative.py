class PositiveNegativePrompt:
    def __init__(self, prompt_path="./prompt/positive_negative.xml"):
        self.prompt_path = prompt_path
        self.prompt = self.load_prompt()

    def load_prompt(self):
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def format_prompt(self, conversation_input):
        return self.prompt.replace("{{CONVERSATION_INPUT}}", str(conversation_input))

    def get_prompt(self):
        return self.prompt
