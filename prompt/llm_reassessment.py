class LLMReassessmentPrompt:
    def __init__(self, prompt_path="./prompt/llm_reassessment.xml"):
        self.prompt_path = prompt_path
        self.prompt = self.load_prompt()

    def load_prompt(self):
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def format_prompt(self, dialogue_sequence, current_prediction, consecutive_range,
                      previous_context=None, consecutive_context=None, next_context=None):
        formatted_prompt = self.prompt

        # Replace dialogue sequence
        dialogue_str = "\n".join([f"{i}: {utt}" for i, utt in enumerate(dialogue_sequence)])
        formatted_prompt = formatted_prompt.replace("{{DIALOGUE_SEQUENCE}}", dialogue_str)

        # Replace current prediction
        formatted_prompt = formatted_prompt.replace("{{CURRENT_PREDICTION}}", str(current_prediction))

        # Replace consecutive range
        formatted_prompt = formatted_prompt.replace("{{CONSECUTIVE_RANGE}}", str(consecutive_range))

        # Replace context information
        if previous_context:
            prev_str = "\n".join([f"{i}: {utt}" for i, utt in enumerate(previous_context)])
            formatted_prompt = formatted_prompt.replace("{{PREVIOUS_CONTEXT}}", prev_str)
        else:
            formatted_prompt = formatted_prompt.replace("{{PREVIOUS_CONTEXT}}", "No previous context")

        if consecutive_context:
            consec_str = "\n".join([f"{i}: {utt}" for i, utt in enumerate(consecutive_context)])
            formatted_prompt = formatted_prompt.replace("{{CONSECUTIVE_CONTEXT}}", consec_str)
        else:
            formatted_prompt = formatted_prompt.replace("{{CONSECUTIVE_CONTEXT}}", "No consecutive context")

        if next_context:
            next_str = "\n".join([f"{i}: {utt}" for i, utt in enumerate(next_context)])
            formatted_prompt = formatted_prompt.replace("{{NEXT_CONTEXT}}", next_str)
        else:
            formatted_prompt = formatted_prompt.replace("{{NEXT_CONTEXT}}", "No next context")

        return formatted_prompt
