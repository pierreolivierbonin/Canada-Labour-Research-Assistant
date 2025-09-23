from collections import OrderedDict
from typing import Any, Dict, List, Literal

from ollama import chat, ChatResponse
from pydantic import BaseModel, ValidationError, Field, model_serializer


class UserInput(BaseModel):
    '''This is the actual data received.'''
    reference_chunk: str = Field(description="The reference chunk to be compared with other target chunks.")
    target_chunk: str = Field(description="The target chunk against which to compare the reference chunk. ")

# class used as a reference by the model
class ComparisonEvaluator(UserInput):
    # reference_chunk: str = Field(..., description="The reference chunk to compare with target chunks.")
    # target_chunk: str = Field(..., description="The target chunk with which the refernce is being compared.")
    category: Literal[
        'very high overlap', 'high overlap', 'medium overlap', 'low overlap', 'no overlap'
        ] = Field(..., description="Comparison result category.")
    reasons: str = Field(..., description="Reasoning explaining why, including:\n* query focus\n* passage focus.")
    summary: str = Field(..., description="A conclusion summarizing the findings.")
    tags: List[str] = Field(..., description="Relevant keywords related to the content of both the reference chunk and target chunk.")

    @model_serializer(when_used='json')
    def sort_model(self) -> Dict[str, Any]:
        # return dict(sorted(self.model_dump().items()))
        return {
            "reference_chunk": self.reference_chunk,
            "target_chunk": self.target_chunk,
            "category": self.category,
            "reasons": self.reasons,
            "summary": self.summary,
            "tags": self.tags
        }

def call_llm(prompt, client=chat, model='gpt-oss:20b', hyperparams=None):
    response = client(model=model, 
                      messages=[
                                {
                                    'role': 'user',
                                    'content': prompt,
                                },
                                ],
                                **hyperparams)
    return response.message.content

def validate_with_model(data_model, llm_response, valid_user_input, verbose=False):
    try:
        llm_response = data_model(**llm_response)
        llm_response.model_dump()
        llm_response.reference_chunk = valid_user_input.reference_chunk
        llm_response.target_chunk = valid_user_input.target_chunk
        
        print("\n\nStructured output first attempt passed validation.")
        if verbose:
            print(llm_response.model_dump_json(indent=4))
        return llm_response, None
    except ValidationError as e:
        print(f"error validating data: {e}")
        error_message = (
            f"This response generated a validation error: {e}."
        )
        return None, error_message

def create_retry_prompt(
    original_prompt, original_response, error_message
):
    retry_prompt = f"""
This is a request to fix an error in the structure of an llm_response.
Here is the original request:
<original_prompt>
{original_prompt}
</original_prompt>

Here is the original llm_response:
<llm_response>
{original_response}
</llm_response>

This response generated an error: 
<error_message>
{error_message}
</error_message>

Compare the error message and the llm_response and identify what 
needs to be fixed or removed
in the llm_response to resolve this error. 

Respond ONLY with valid JSON. Do not include any explanations or 
other text or formatting before or after the JSON string.
"""
    return retry_prompt


def rectify_llm_response(
    prompt, data_model, n_retry=5, model='gemma3n:latest'
):
    # Initial LLM call
    response_content = call_llm(prompt, model=model)
    current_prompt = prompt

    # Try to validate with the model
    # attempt: 0=initial, 1=first retry, ...
    for attempt in range(n_retry + 1):
        print(f"\nData model validation attempt #{attempt+1}\n")
        validated_data, validation_error = validate_with_model(
            data_model, response_content
        )

        if validation_error:
            if attempt < n_retry:
                print(f"retry {attempt} of {n_retry} failed, trying again...")
            else:
                print(f"Max retries reached. Last error: {validation_error}")
                return None, (
                    f"Max retries reached. Last error: {validation_error}"
                )

            validation_retry_prompt = create_retry_prompt(
                original_prompt=current_prompt,
                original_response=response_content,
                error_message=validation_error
            )
            response_content = call_llm(
                validation_retry_prompt, model=model
            )
            current_prompt = validation_retry_prompt
            continue

        # If you get here, both parsing and validation succeeded
        return validated_data, None


if __name__ == "__main__":


    test_user_data = '''
    {
    "reference_chunk": "Section 247.95 (1) If, during a leave of absence that is taken under this Division, the wages or benefits of the group of employees of which an employee is a member are changed as part of a plan to reorganize the industrial establishment in which that group is employed, the employee is entitled, on reinstatement under this section, to receive the wages and benefits in respect of that employment that that employee would have been entitled to receive had that employee been working when the reorganization took place. Marginal note: Notice of change in wages or benefits (2) The employer of an employee who is on leave and whose wages or benefits would be changed as a result of the reorganization shall, as soon as practicable, send a notice to the employee at their last known address. 2008, c. 15, s. 1 Marginal note: Prohibition — employee Section 247.96 (1) No employer may dismiss, suspend, lay off, demote or discipline an employee because they are a member of the reserve force or intend to take or have taken a leave of absence under this Division or take into account the fact that an employee is a member of the reserve force or intends to take or has taken a leave of absence under this Division in a decision to promote or train them. Marginal note: Prohibition — future employee (2) No person may refuse to employ a person because they are a member of the reserve force. 2008, c. 15, s. 1 Marginal note: Regulations",
    "target_chunk": "Section 327 (1) The authorized representative of a Canadian vessel shall ensure that no crew member is required to make an advance payment at the beginning of their employment towards the expenses referred to in subsection 94(1) of the Act or section 328. (2) The authorized representative shall ensure that the time a crew member spends waiting to be returned and being returned under subsection 94(1) of the Act or section 328 is not deducted from the paid leave accrued to them."
    }
    '''
    
    test_validation = UserInput.model_validate_json(test_user_data) # returns the validated Pydantic model if validation was successful
    test_validation.model_dump_json(indent=2)
    
    # test with a different example of matched passages
    new_test = '''{"reference_chunk": "Section 247.95 (1) If, during a leave of absence that is taken under this Division, the wages or benefits of the group of employees of which an employee is a member are changed as part of a plan to reorganize the industrial establishment in which that group is employed, the employee is entitled, on reinstatement under this section, to receive the wages and benefits in respect of that employment that that employee would have been entitled to receive had that employee been working when the reorganization took place. Marginal note: Notice of change in wages or benefits (2) The employer of an employee who is on leave and whose wages or benefits would be changed as a result of the reorganization shall, as soon as practicable, send a notice to the employee at their last known address. 2008, c. 15, s. 1 Marginal note: Prohibition — employee Section 247.96 (1) No employer may dismiss, suspend, lay off, demote or discipline an employee because they are a member of the reserve force or intend to take or have taken a leave of absence under this Division or take into account the fact that an employee is a member of the reserve force or intends to take or has taken a leave of absence under this Division in a decision to promote or train them. Marginal note: Prohibition — future employee (2) No person may refuse to employ a person because they are a member of the reserve force. 2008, c. 15, s. 1 Marginal note: Regulations",
    "target_chunk": "Section 71 Notwithstanding section 67, when a designated employee is entitled to severance pay from the Corporation pursuant to a collective agreement, an arbitral award or terms and conditions of employment, the period for which the designated employee is entitled to severance pay is deemed not to include any period of employment for which the designated employee is entitled to severance pay under section 70. Marginal note: Deemed lay-off"}'''
    
    new_test2 = '''{"reference_chunk": "**Waiting time** “Waiting time” applies mostly in the trucking industry. “Working hours” usually means all hours from the time that a motor vehicle operator begins their work shift until the time they are relieved of their job responsibilities. It does not include certain times:  during a work shift when they are relieved of their job responsibilities. This includes authorized meals, rest and other wait time while en route or at their destination spent during stops en route due to illness or fatigue resting en route as one of 2 operators of a motor vehicle that is fitted with a sleeper berth resting while en route in a motel, hotel or other similar regular place of rest where sleeping accommodation is provided  Generally, “waiting time” is not considered to be “hours of work” to be paid.  **Lay-over time** This period is a common occurrence in the road transportation industry. A “lay-over” occurs when a driver completes their delivery and is awaiting further instructions regarding other possible pick-ups. The period that the driver is out of service is a lay-over and this is not considered to be hours of work.  **Related links**  Hours of work - Federally regulated workplaces  **Page details**  Date modified: 2023-07-25",
    "target_chunk": "Section 39 (1) No motor carrier shall request, require or allow a driver to drive and no driver shall drive after the driver has accumulated more than 15 hours of driving time or 18 hours of on-duty time unless they take at least 8 consecutive hours of off-duty time before driving again. (2) No motor carrier shall request, require or allow a driver to drive and no driver shall drive if more than 20 hours of time has elapsed between the conclusion of the most recent period of 8 or more consecutive hours of off-duty time and the beginning of the next period of 8 or more consecutive hours of off-duty time."}'''

    new_test3 = '''{"reference_chunk": "test one two test",
    "target_chunk": "test two three test"}'''

    validated_test = UserInput.model_validate_json(new_test2) # change this to impact everything else below

    # Create prompt with user data and expected JSON structure
    prompt = [
        {"role":"system",
         "content": "You are an expert at comparing legal, regulatory, and policy documents. Your task is to accurately compare two documents and extract key information based on the user-provided schema.",
        },
        {"role": "user",
         "content":f"Please compare the following documents:\n\n{validated_test.model_dump_json(indent=2)}"
        }
        ]

    # call the LLM to produce structured output following the data model schema specified above
    print(f"\nRequested output format:\n\n{ComparisonEvaluator.model_json_schema()}")
    response = chat(model="gemma3n:latest", messages=prompt, format=ComparisonEvaluator.model_json_schema())
    validated = ComparisonEvaluator.model_validate_json(response.message.content)
    print(response["message"]["content"])