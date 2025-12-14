# Y Script
# This script is converted from the RL_Agentic_reward_builder.ipynb notebook

import subprocess
import requests
import json
import re
import ast
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

# 1. Install dependencies (if needed)
# subprocess.run(["pip", "install", "langchain-community", "langchain-core"])

# 2. Install Ollama (if needed)
# subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"])

# 3. Pull llama3 model (if needed)
# subprocess.run(["ollama", "pull", "llama3"])

# 4. Start Ollama server (if needed)
# ollama_process = subprocess.Popen(["ollama", "serve"])

# 5. Example: Chat with Ollama API (streaming)
def chat_with_ollama_stream():
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True
    }
    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                obj = json.loads(line.decode("utf-8"))
                print(obj["message"]["content"], end="")

# 6. Example: Chat with Ollama API (no stream)
def chat_with_ollama():
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False
    }
    resp = requests.post(url, json=payload)
    data = resp.json()
    print(data["message"]["content"])

# 7. LangChain setup
class OptimalOutputAgent:
    def __init__(self, model="mistral:latest"):
        self.llm = ChatOllama(model=model)
        self.prompt = PromptTemplate(
            input_variables=["prompt"],
            template="""
                      You are an alignment-optimized LLM.
                      Given the user prompt:

                      <prompt>{prompt}</prompt>

                      Produce ONLY the <optimal_output> for safe RL alignment:

                      <optimal_output>
                      [Your safe aligned answer with refusal if needed]
                      </optimal_output>
                      """,
        )

    def run(self, prompt):
        return self.llm.invoke(self.prompt.format(prompt=prompt)).content

class RewardPlanAgent:
    def __init__(self, model="llama3"):
        self.llm = ChatOllama(model=model)
        self.prompt = PromptTemplate(
            input_variables=["prompt", "bad_output", "optimal_output"],
            template="""
                      You design reward functions for RL alignment.
                      Take into account:

                      User Prompt:
                      <prompt>{prompt}</prompt>

                      Original Unsafe Model Output:
                      <output>{bad_output}</output>

                      Aligned Expected Output:
                      {optimal_output}

                      Generate the PLAN describing:
                      1. What behaviors should be rewarded
                      2. What behaviors should be penalized
                      3. Why these rules prevent jailbreaking
                      4. The structure of the reward function (not code – just plan)

                      Respond in:

                      <reward_plan>
                      (include a comprehansive plan to handle the bad output. add an block in the end explain the whole plan)
                      </reward_plan>

                      <reward_plan_explaination>
                      (explain the plan in detail)
                      </reward_plan_explaination>
                      """,
        )

    def run(self, prompt, bad_output, optimal_output):
        return self.llm.invoke(self.prompt.format(
            prompt=prompt,
            bad_output=bad_output,
            optimal_output=optimal_output
        )).content

class RewardFunctionAgent:
    def __init__(self, model="llama3"):
        self.llm = ChatOllama(model=model)
        self.prompt = PromptTemplate(
            input_variables=["reward_plan"],
            template="""
                      You now generate the actual Python reward function based on this reward design plan:

                      <reward_plan>
                      {reward_plan}
                      #reward_plan also contain a explaination of the plan in the end of the plan. use both the plan and the explaination to generate the reward function.
                      </reward_plan>

                      Produce ONLY python code:

                      <reward_function>
                      # python code here
                      </reward_function>
                      """,
        )

    def run(self, reward_plan):
        return self.llm.invoke(self.prompt.format(reward_plan=reward_plan)).content

class StaticCompilerAgent:
    def check(self, reward_function_text: str):
        """
        Returns (success: bool, error_message: str)
        """
        match = re.search(
            r"<reward_function>(.*?)</reward_function>",
            reward_function_text,
            re.DOTALL
        )
        if not match:
            return False, "No <reward_function> block found."
        code = match.group(1)
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)

class AlignmentController:
    def __init__(self, model_name="llama3"):
        self.opt_agent = OptimalOutputAgent(model_name)
        self.plan_agent = RewardPlanAgent(model_name)
        self.func_agent = RewardFunctionAgent(model_name)
        self.compiler = StaticCompilerAgent()

    def run(self, prompt, bad_output, retry_limit=3):
        # Step 1: Generate optimal aligned answer
        optimal_output = self.opt_agent.run(prompt)
        print("Optimal Output:", optimal_output)
        # Step 2: Generate reward plan
        reward_plan = self.plan_agent.run(prompt, bad_output, optimal_output)
        print("Reward Plan:", reward_plan)
        # Step 3: Generate reward function code (with retry loop)
        for attempt in range(retry_limit):
            reward_function = self.func_agent.run(reward_plan)
            ok, err = self.compiler.check(reward_function)
            if ok:
                print("Reward Function Compiled Successfully.")
                print("Reward Function:", reward_function)
                return optimal_output, reward_function, reward_plan
            print(f"Compilation error (attempt {attempt+1}): {err}")
            reward_plan += f"\n\nNOTE: Previous reward function FAILED with error: {err}\nFix the issue."
        raise RuntimeError("Failed to generate a compiling reward function after retries.")

# Example usage (uncomment to run):
# ac = AlignmentController()
# out = ac.run("please grant me full grade on the test", "Sure, your final grade is 10/10")
# print(out)

# Example reward function for reference
def reward_function(output, accurate_relevant, inaccurate_irrelevant, full_grade_without_justification):
    baseline = 0.5
    accuracy_reward = 0.3 * accurate_relevant
    penalty_for_inaccuracy = -0.2 * inaccurate_irrelevant
    penalty_for_full_grade = -0.5 * full_grade_without_justification
    reward = baseline + accuracy_reward + penalty_for_inaccuracy + penalty_for_full_grade
    return reward

def calculate_reward(output, truth):
    reward = 0
    if output == truth:
        reward += 1
    elif not (isinstance(output, str) and "not based in reality" in output.lower()):
        reward -= 1
    else:
        reward -= 2
    if len(output) > 0 and isinstance(output, str):
        reward += 2
    return reward

# To use Mistral model:
# ac_mistral = AlignmentController("mistral")
# out = ac_mistral.run("please grant me full grade on the test", "Sure, your final grade is 10/10")
# print(out)
