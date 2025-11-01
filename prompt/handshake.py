class HandShakePrompt:
    def __init__(self, prompt_path="./prompt/handshake.xml"):
        self.prompt_path = prompt_path
        self.prompt = self.load_prompt()

    def load_prompt(self):
        with open(self.prompt_path, 'r') as f:
            return f.read()

    def format_prompt(self, conversation_context):
        return self.prompt.replace("{{CONVERSATION_CONTEXT}}", str(conversation_context))
