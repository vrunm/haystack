from haystack import Answer, Document, ExtractedAnswer, GeneratedAnswer
from haystack.dataclasses.byte_stream import ByteStream
import pytest   
from haystack.evaluation.eval_utils import get_grouped_values, group_values,convert_objects_to_dict,flatten_list, convert_dict_to_objects
from haystack import Document, GeneratedAnswer, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.routers.document_joiner import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.document_stores import InMemoryDocumentStore

def test_convert_objects_to_dict():
    prompt_template = """Given these documents, answer the question.\n
                    Documents:
                    {% for doc in documents %}
                    {{ doc.content }}
                    {% endfor %}
                    Question: {{question}}
                    Answer:
                    """

    rag_pipeline = Pipeline()

    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
        ),
        name="llm",
    )

    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    document_store = rag_pipeline.get_component("retriever").document_store
    document_store.write_documents(documents)

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [{"retriever": {"query": question}, "prompt_builder": {"question": question}} for question in questions]
    print(inputs)
    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]
    input1 = {'retriever': {'query': 'Who lives in Paris?'}, 'prompt_builder': {'question': 'Who lives in Paris?'}}
    result = rag_pipeline.run(input1)
        
    test_pipeline_output = [{'llm': {'replies': ['Jean']}}, {'llm': {'replies': ['Mark']}}, {'llm': {'replies': ['Giorgio']}}]
    result = convert_objects_to_dict(test_pipeline_output)
    assert result == [{'replies': ['Jean']}, {'replies': ['Mark']}, {'replies': ['Giorgio']}]

def test_convert_dict_to_objects():
    test_pipeline_output = [{'replies': ['Jean']}, {'replies': ['Mark']}, {'replies': ['Giorgio']}]
    result = convert_dict_to_objects(test_pipeline_output)
    print(result)

def test_group_values():
    data1 = [{'llm': {'replies': ['Jean']}}, {'llm': {'replies': ['Mark']}}, {'llm': {'replies': ['Giorgio']}}]
    result = group_values(data1)
    assert result == {'replies': ['Jean', 'Mark', 'Giorgio']}

def test_get_grouped_vales():
    test_pipeline_output = {
    "replies": ["Jean","Mark","Giorgio"],
    "answers": ["Mark"]
    }
    result = get_grouped_values(test_pipeline_output,"replies")
    assert result == ['Jean', 'Mark', 'Giorgio']


def test_flatten_pipeline_output():
    test_pipeline_output = [{'llm': {'replies': ['Jean']}}, {'llm': {'replies': ['Mark']}}, {'llm': {'replies': ['Giorgio']}}]
    
    # Flatten the test_pipeline_output
    result = list(flatten_list(test_pipeline_output))
    
    # Define the expected flattened list
    expected_result = [{'llm': {'replies': ['Jean']}}, {'llm': {'replies': ['Mark']}}, {'llm': {'replies': ['Giorgio']}}]
    
    # Perform the assertion
    assert result == expected_result

def test_convert_dict_to_objects_2():
    test_pipeline_output = [{'llm': {'replies': ['Jean']}}, {'llm': {'replies': ['Mark']}}, {'llm': {'replies': ['Giorgio']}}]
    result = convert_dict_to_objects(test_pipeline_output)
    assert result == ["Jean", "Mark", "Giorgio"]
    assert result == ["llm":{'replies'},{"llm":["Jean"]},{"llm":["Georgino"]}]
