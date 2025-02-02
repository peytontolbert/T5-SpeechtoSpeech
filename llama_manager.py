from transformers import pipeline
from openai import OpenAI

class LlamaManager:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype="bfloat16",
            device_map="cuda:0",
            max_new_tokens=800
        )
    def generate_response(self, conversation_history, system_prompt):
        # Prepare user input with the last 20 messages from conversation history
        user_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-20:]])
        
        # Generate a response from LLaMA
        #outputs = self.pipe(
        #    prompt,
        #    max_new_tokens=800
        #)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800
        )
        

        generated_text = response.choices[0].message.content
        return generated_text

    def parse_ai_response(self, generated_text, participants):
        # Simple parsing logic; can be improved with regex or more advanced NLP
        lines = generated_text.strip().split('\n')
        ai_responses = []
        for line in lines:
            if line.startswith("You:"):
                continue  # Skip user lines
            if ':' in line:
                name, content = line.split(':', 1)
                name = name.strip()
                content = content.strip()
                # Parse action if present
                if '(' in content and ')' in content:
                    message, action_part = content.split('(', 1)
                    message = message.strip()
                    action_part = action_part.strip(')')
                    if action_part.startswith("action:"):
                        action_details = action_part[len("action:"):].strip()
                        if action_details.startswith("interrupt"):
                            parts = action_details.split()
                            action = "interrupt"
                            time_wait = parts[1] if len(parts) > 1 else "0s"
                            ai_responses.append({
                                'name': name,
                                'content': message,
                                'action': action,
                                'time': time_wait
                            })
                        else:
                            ai_responses.append({
                                'name': name,
                                'content': message,
                                'action': action_details
                            })
                    else:
                        ai_responses.append({'name': name, 'content': content})
                else:
                    ai_responses.append({'name': name, 'content': content})
        return ai_responses
