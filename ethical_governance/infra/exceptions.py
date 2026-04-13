"""
infra/exceptions.py — Eccezioni di dominio per Ethical Governance Platform v2.7.

Queste eccezioni permettono ai moduli core/governance/ml di segnalare errori
di dominio in modo esplicito, lasciando al layer API la traduzione in HTTP.
"""
from __future__ import annotations


class DomainError(Exception):
    """Base class per tutti gli errori di dominio della piattaforma."""


class FeatureMissingError(DomainError):
    """Alcune feature richieste dal modello sono mancanti."""


class ModelNotFoundError(DomainError):
    """Il modello richiesto non esiste nel registry o su disco."""


class ReferenceDataMissingError(DomainError):
    """Non sono disponibili dati di riferimento sufficienti."""


class PredictionNotFoundError(DomainError):
    """La prediction richiesta non è presente nell'audit trail."""


class FeedbackOwnershipError(DomainError):
    """Il feedback appartiene a un tenant diverso da quello richiedente."""


class TaskNotFoundError(DomainError):
    """Il task HITL richiesto non è stato trovato."""


class InvalidComparisonRequestError(DomainError):
    """La richiesta di confronto modelli non è valida."""


class RetrainingRejectedError(DomainError):
    """Il retraining non può essere avviato per lo stato corrente."""
