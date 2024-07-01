import boto3
import json
from typing import Dict
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
import time
#from langchain.chains.question_answering import load_qa_chain
#from langchain_community.llms import SagemakerEndpoint
#from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
#from langchain_core.prompts import PromptTemplate
#from langchain_core.documents import Document


sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
sagemaker_role = "AmazonSageMaker-ExecutionRole-20240618T160945"
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name = region)
endpoint_name = "spectra-test1"


# model_id = "huggingface-llm-mistral-7b"
# model = JumpStartModel (model_id=model_id, region=region, role=sagemaker_role)
# example_payloads = model.retrieve_all_examples()
#print(example_payloads)

question = "Write a program to compute factorial in python:"
input_data = {
    'inputs': f'{question}', 
    'parameters': {
        'max_new_tokens': 200, 
        'decoder_input_details': True, 
        'details': True
    }
}

body = json.dumps(input_data)
response = sagemaker_runtime.invoke_endpoint(
    EndpointName = endpoint_name,
    ContentType = 'application/json',
    Accept='application/json',
    Body = body
)


# example_doc_1 = """
# Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital.
# Since she was diagnosed with a brain injury, the doctor told Peter to stay besides her until she gets well.
# Therefore, Peter stayed with her at the hospital for 3 days without leaving.
# """

# docs = [
#     Document(
#         page_content=example_doc_1,
#     )
# ]

# prompt_template = """Use the following pieces of context to answer the question at the end.

# {context}

# Question: {question}
# Answer:"""
# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )

# class ContentHandler(LLMContentHandler):
#     content_type = "application/json"
#     accepts = "application/json"

#     def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
#         input_str = json.dumps(input_data)
#         return input_str.encode("utf-8")

#     def transform_output(self, output: bytes) -> str:
#         response_json = json.loads(output.read().decode("utf-8"))
#         return response_json[0]["generated_text"]

# content_handler = ContentHandler()

# chain = load_qa_chain(
#     llm=SagemakerEndpoint(
#         endpoint_name=endpoint_name,
#         client=sagemaker_runtime,
#         model_kwargs={"temperature": 0.9},
#         content_handler=content_handler,
#     )
# )

# query = """How long was Elizabeth hospitalized?
# """

# chain({"input_documents": docs, "question": query}, return_only_outputs=True)

result = json.loads(response['Body'].read())
result[0]['generated_text']
