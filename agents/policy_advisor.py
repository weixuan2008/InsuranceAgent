from utils.config import get_llm
from utils.prompts import POLICY_ADVISOR_PROMPT

class PolicyAdvisorAgent:
    def __init__(self):
        self.llm = get_llm() 
    
    def recommend_policy(self, customer_profile):
        prompt = POLICY_ADVISOR_PROMPT.format(customer_profile=customer_profile)
        print("Calling Agent: PolicyAdvisorAgent")
        response = self.llm.invoke(prompt)
        llm_response_content = response.content if hasattr(response, "content") else str(response)
        return llm_response_content