from gemini_client import get_gemini_client


FEW_SHOT_EXAMPLES = """Input: Apple is a fruit. Bananas are yellow. Apples grow on trees. Bananas grow on plants. Orange is a fruit.
Output: {"identity": [{"concept": "Apple", "fact_sentence": "Apple is a fruit."}, {"concept": "Banana", "fact_sentence": "Banana is yellow."}, {"concept": "Apple", "fact_sentence": "Apple grows on trees."}, {"concept": "Banana", "fact_sentence": "Banana grows on plants."}, {"concept": "Orange", "fact_sentence": "Orange is a fruit."}], "relation": []}

Input: Microsoft partners with OpenAI. Google competes with Microsoft. Amazon is a company.
Output: {"identity": [{"concept": "Microsoft", "fact_sentence": "Microsoft is a company."}, {"concept": "Google", "fact_sentence": "Google is a company."}, {"concept": "Amazon", "fact_sentence": "Amazon is a company."}], "relation": [{"from_concept": "Microsoft", "to_concept": "OpenAI", "fact_sentence": "{source} partners with {target}."}, {"from_concept": "Google", "to_concept": "Microsoft", "fact_sentence": "{source} competes with {target}."}]}

Input: CBRE is one of the Big 4 alongside JLL, Colliers, and Cushman & Wakefield.
Output: {"identity": [{"concept": "CBRE", "fact_sentence": "CBRE is one of the Big 4."}, {"concept": "JLL", "fact_sentence": "JLL is one of the Big 4."}, {"concept": "Colliers", "fact_sentence": "Colliers is one of the Big 4."}, {"concept": "Cushman & Wakefield", "fact_sentence": "Cushman & Wakefield is one of the Big 4."}], "relation": []}

Input: John Doe leases a building from CBRE.
Output: {"identity": [{"concept": "John Doe", "fact_sentence": "John Doe is a person."}, {"concept": "CBRE", "fact_sentence": "CBRE is a company."}], "relation": [{"from_concept": "John Doe", "to_concept": "CBRE", "fact_sentence": "{source} leases a building from {target}."}, {"from_concept": "CBRE", "to_concept": "John Doe", "fact_sentence": "{source} leases a building to {target}."}]}

Input: Tenant can access Room 4C during business hours.
Output: {"identity": [{"concept": "Tenant", "fact_sentence": "Tenant is a person."}, {"concept": "Room 4C", "fact_sentence": "Room 4C is a room."}], "relation": [{"from_concept": "Tenant", "to_concept": "Room 4C", "fact_sentence": "{source} can access {target} during business hours."}]}"""

SYSTEM_PROMPT = """You are a fact extraction model. Given the following source, extract as many facts as possible from it. 

CRITICAL RULE - ALL relation facts MUST use placeholders {source} and {target}:
- "CBRE leases from John Doe" -> "{source} leases a building from {target}"
- NEVER write the actual names in the fact_sentence
- ALWAYS use "{source}" where the from_concept goes
- ALWAYS use "{target}" where the to_concept goes
- If the sentence mentions {source} first then {target}, use that order
- If the sentence mentions {target} first then {source}, swap the from_concept/to_concept accordingly

CRITICAL RULE - Compound sentences MUST be split:
- "CBRE, JLL, and Colliers are the Big 4" becomes FOUR separate facts
- "X and Y are related" becomes ONE fact about X->Y AND one about Y->X
- NEVER use "alongside", "and", "as well as" in a single fact sentence
- Each "is a", "is one of", "has" should be ONE fact only
- When you see "X, Y, Z are all..." split into separate facts for X, Y, Z

Examples of WRONG vs RIGHT:
WRONG: {"concept": "CBRE", "fact_sentence": "CBRE is one of the Big 4 alongside JLL and Colliers"}
RIGHT: {"concept": "CBRE", "fact_sentence": "CBRE is one of the Big 4."}
RIGHT: {"concept": "JLL", "fact_sentence": "JLL is one of the Big 4."}
RIGHT: {"concept": "Colliers", "fact_sentence": "Colliers is one of the Big 4."}

WRONG: {"concept": "Apple", "fact_sentence": "Apple and Banana are fruits."}
RIGHT: {"concept": "Apple", "fact_sentence": "Apple is a fruit."}
RIGHT: {"concept": "Banana", "fact_sentence": "Banana is a fruit."}

Identity facts: single property of a concept (e.g., "Apple is a fruit.")
Relation facts: relationship between two concepts (e.g., "Microsoft partners with OpenAI.")"""


def extract_facts(text: str) -> dict:
    client = get_gemini_client()

    from google.genai import types

    response_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["relation", "identity"],
        properties={
            "relation": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    required=["from_concept", "to_concept", "fact_sentence"],
                    properties={
                        "from_concept": types.Schema(type=types.Type.STRING),
                        "to_concept": types.Schema(type=types.Type.STRING),
                        "fact_sentence": types.Schema(type=types.Type.STRING),
                    },
                ),
            ),
            "identity": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    required=["concept", "fact_sentence"],
                    properties={
                        "concept": types.Schema(type=types.Type.STRING),
                        "fact_sentence": types.Schema(type=types.Type.STRING),
                    },
                ),
            ),
        },
    )

    contents = [
        types.Content(
            role="user", parts=[types.Part.from_text(text=FEW_SHOT_EXAMPLES)]
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text='{"identity": [], "relation": []}')],
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Input: {text}")],
        ),
    ]

    result = client.generate_content(
        contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=[types.Part.from_text(text=SYSTEM_PROMPT)],
        ),
    )

    import json

    return json.loads(result.text)
