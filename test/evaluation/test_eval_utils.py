import pytest
from haystack import (Answer, Document, ExtractedAnswer, GeneratedAnswer,
                      Pipeline)
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers import (InMemoryBM25Retriever,
                                            InMemoryEmbeddingRetriever)
from haystack.dataclasses.byte_stream import ByteStream
from haystack.document_stores import InMemoryDocumentStore
from haystack.evaluation.eval import EvaluationResult, eval
from haystack.evaluation.eval_utils import (convert_dict_to_objects,
                                            convert_objects_to_dict,
                                            create_grouped_values,
                                            flatten_list,
                                            retrieve_grouped_values)
from haystack.evaluation.metrics import Metric


def test_convert_objects_to_dict():
    """
    Test to check the the component serialization for the RAG pipeline.
    The test asserts the output of the eval has been serialized correctly 
    The RAG pipeline consists
    """
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

    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]
    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)
    result = convert_objects_to_dict(eval_result.outputs)
    assert result == [{"replies": ["Jean"]}, {"replies": ["Mark"]}, {"replies": ["Giorgio"]}]

def test_convert_dict_to_objects():
    """
    Test to check the component deserialization for the RAG pipeline.
    """
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
    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)
    result = convert_dict_to_objects(eval_result.outputs)
    assert result == [{"replies": ["Jean"]}, {"replies": ["Mark"]}, {"replies": ["Giorgio"]}]


def test_create_grouped_values():
    """
    Test to check the desrialization of the  RAG pipeline
    """
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
    result = rag_pipeline.run(inputs[0])
    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]
    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)
    result = create_grouped_values(eval_result.outputs)
    assert result == {"replies": ["Jean", "Mark", "Giorgio"]}


def test_retrieve_grouped_values():
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
    result = rag_pipeline.run(inputs[0])
    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]
    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)
    result = create_grouped_values(eval_result.outputs)
    grouped_values = retrieve_grouped_values(result, "replies")
    assert grouped_values == ["Jean", "Mark", "Giorgio"]

def test_flatten_list():
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
    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]
    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)
    result = list(flatten_list(eval_result.outputs))
    print(result)
    #Assert the result of the flattened is the list required to be 
    #The flattened
    assert result == [{"llm": {"replies": ["Jean"]}}, {"llm": {"replies": ["Mark"]}}, {"llm": {"replies": ["Giorgio"]}}]
