"""Tests for the Bitext support-dataset -> knowledge-base/SFT builder.

Uses small fixture records (no network/dataset download) to verify the
transformation logic independently of `load_support_dataset`.
"""
import json

from src.data_engineering.ingestion import load_documents_from_directory
from src.data_engineering.support_dataset_builder import (
    build,
    build_knowledge_base,
    build_sft_dataset,
    derive_escalation,
    derive_priority,
)

FIXTURE_RECORDS = [
    {
        "instruction": "I need to cancel order 12345",
        "category": "ORDER",
        "intent": "cancel_order",
        "response": "Short answer.",
    },
    {
        "instruction": "please cancel my order asap",
        "category": "ORDER",
        "intent": "cancel_order",
        "response": "I understand you want to cancel your order. Here are the steps to do so.",
    },
    {
        "instruction": "where is my refund",
        "category": "REFUND",
        "intent": "track_refund",
        "response": "Refunds are processed within 5-10 business days.",
    },
    {
        "instruction": "I want to speak to a human agent",
        "category": "CONTACT",
        "intent": "contact_human_agent",
        "response": "Connecting you to a human agent now.",
    },
]


def test_derive_priority_maps_keywords():
    assert derive_priority("REFUND", "track_refund") == "high"
    assert derive_priority("ORDER", "cancel_order") == "high"  # "cancel" keyword
    assert derive_priority("DELIVERY", "delivery_period") == "medium"
    assert derive_priority("OTHER", "store_location") == "low"


def test_derive_escalation_flags_complaint_and_human_agent():
    assert derive_escalation("contact_human_agent") is True
    assert derive_escalation("complaint") is True
    assert derive_escalation("track_order") is False


def test_build_knowledge_base_dedupes_by_intent_keeping_longest_response():
    kb = build_knowledge_base(FIXTURE_RECORDS)
    doc_ids = {doc["doc_id"] for doc in kb}
    assert doc_ids == {"cancel_order", "track_refund", "contact_human_agent"}

    cancel_doc = next(doc for doc in kb if doc["doc_id"] == "cancel_order")
    assert cancel_doc["text"] == "I understand you want to cancel your order. Here are the steps to do so."
    assert cancel_doc["metadata"] == {"category": "ORDER", "intent": "cancel_order"}


def test_build_sft_dataset_grounds_context_in_knowledge_base():
    kb = build_knowledge_base(FIXTURE_RECORDS)
    sft = build_sft_dataset(FIXTURE_RECORDS, kb)
    assert len(sft) == len(FIXTURE_RECORDS)

    example = sft[0]
    assert example["query"] == "I need to cancel order 12345"
    # Context is the canonical (longest) response for the intent, not
    # necessarily this record's own (possibly shorter) response.
    assert example["context"] == "I understand you want to cancel your order. Here are the steps to do so."

    parsed_response = json.loads(example["response"])
    assert parsed_response["intent"] == "cancel_order"
    assert parsed_response["priority"] == "high"
    assert parsed_response["escalation_required"] is False


def test_knowledge_base_output_is_loadable_by_ingestion(tmp_path):
    kb = build_knowledge_base(FIXTURE_RECORDS)
    (tmp_path / "bitext_knowledge_base.json").write_text(json.dumps(kb), encoding="utf-8")

    documents = load_documents_from_directory(tmp_path)
    assert {d.doc_id for d in documents} == {"cancel_order", "track_refund", "contact_human_agent"}


def test_build_writes_knowledge_base_and_sft_files(tmp_path, monkeypatch):
    import src.data_engineering.support_dataset_builder as builder_module

    monkeypatch.setattr(
        builder_module, "load_support_dataset", lambda max_samples=None, seed=42: FIXTURE_RECORDS
    )
    build(max_samples=None, output_dir=tmp_path)

    kb_path = tmp_path / "bitext_knowledge_base.json"
    sft_path = tmp_path / "bitext_sft_dataset.json"
    assert kb_path.exists() and sft_path.exists()
    assert len(json.loads(kb_path.read_text())) == 3
    assert len(json.loads(sft_path.read_text())) == len(FIXTURE_RECORDS)
