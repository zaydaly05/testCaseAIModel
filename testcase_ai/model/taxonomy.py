"""Requirement categories the model predicts, and how they map to test design."""

from __future__ import annotations

CATEGORIES: dict[str, str] = {
    "functional": "A feature or behaviour the system must provide",
    "validation": "Rules on input format, required fields or allowed values",
    "boundary": "Numeric limits, lengths, counts or ranges",
    "security": "Authentication, authorization, privacy and data protection",
    "error_handling": "How failures, invalid situations and errors are handled",
    "reliability": "Availability, fault tolerance, backup and recovery",
    "workflow": "States, status transitions, persistence and process order",
    "performance": "Response times, throughput, load and scalability",
    "notification": "Emails, SMS, push or in-app messages sent to users",
    "usability": "UI, look and feel, accessibility and ease of use",
    "compliance": "Legal, regulatory and policy requirements",
    "operational": "Environment, compatibility, portability and maintainability",
}

# Labels used by the PROMISE NFR dataset (and its extended version) -> our categories.
PROMISE_LABELS: dict[str, str] = {
    "F": "functional",
    "FR": "functional",
    "SE": "security",
    "PE": "performance",
    "SC": "performance",
    "US": "usability",
    "LF": "usability",
    "A": "reliability",
    "FT": "reliability",
    "L": "compliance",
    "O": "operational",
    "MN": "operational",
    "PO": "operational",
}

# Base risk (1-3) used to prioritise test cases; higher means test first.
CATEGORY_RISK: dict[str, int] = {
    "functional": 3,
    "validation": 2,
    "boundary": 2,
    "security": 3,
    "error_handling": 2,
    "reliability": 2,
    "workflow": 3,
    "performance": 2,
    "notification": 2,
    "usability": 1,
    "compliance": 2,
    "operational": 1,
}

HIGH_RISK_WORDS = (
    "payment", "pay", "card", "money", "transfer", "refund", "order", "checkout", "password",
    "login", "log in", "sign in", "account", "personal", "delete", "data", "medical", "patient",
    "security", "permission", "must", "shall", "legal", "invoice", "balance",
)
