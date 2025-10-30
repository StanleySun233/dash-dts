class DTSPrompt:
    def __init__(self, prompt_path="./prompt/dts.xml"):
        self.prompt_path = prompt_path
        self.prompt = self.load_prompt()

    def load_prompt(self):
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def format_prompt(self, conversation_context, few_shot_examples=None, similarity_examples=None):
        formatted_prompt = self.prompt.replace("{{CONVERSATION_CONTEXT}}", str(conversation_context))

        if few_shot_examples:
            formatted_prompt = formatted_prompt.replace("{{FEW_SHOT_EXAMPLES}}", str(few_shot_examples))
        else:
            formatted_prompt = formatted_prompt.replace("{{FEW_SHOT_EXAMPLES}}", "No few-shot examples provided")

        if similarity_examples:
            formatted_prompt = formatted_prompt.replace("{{SIMILARITY_EXAMPLES}}", str(similarity_examples))
        else:
            formatted_prompt = formatted_prompt.replace("{{SIMILARITY_EXAMPLES}}", "No similarity examples provided")

        return formatted_prompt
