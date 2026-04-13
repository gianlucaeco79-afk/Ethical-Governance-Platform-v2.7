"""
infra/queue.py — MessageQueueInterface (pattern da V15.5)

ABC che astrae la coda messaggi. MockMessageQueue per sviluppo/test;
sostituire con KafkaMessageQueue o RedisStreamQueue senza toccare
nessun altro modulo.

Sostituzione in produzione (esempio):
    from ethical_governance.infra.queue import MessageQueueBase
    class KafkaMessageQueue(MessageQueueBase):
        async def send_message(self, topic, message): ...
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict

import httpx
import numpy as np

from ethical_governance.infra.observability import get_logger

logger = get_logger(__name__)


class MessageQueueBase(ABC):
    """
    Interfaccia astratta per la coda messaggi.
    Implementazioni concrete: MockMessageQueue, KafkaMessageQueue, ecc.
    """

    @abstractmethod
    async def send_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Invia un messaggio al topic specificato.
        Deve essere idempotente e gestire internamente i retry.
        Solleva RuntimeError dopo l'esaurimento dei tentativi.
        """
        raise NotImplementedError


class MockMessageQueue(MessageQueueBase):
    """
    Coda in-memory con retry esponenziale e simulazione di errori di rete.
    Usare solo in sviluppo e test.
    """

    def __init__(self, max_retries: int) -> None:
        self._max_retries = max_retries
        self._sent: list[Dict[str, Any]] = []

    async def _attempt(self, topic: str, message: Dict[str, Any]) -> None:
        if np.random.rand() < 0.05:
            raise IOError("Errore di rete simulato.")
        await asyncio.sleep(0.001)
        self._sent.append({"topic": topic, "message": message})

    async def send_message(self, topic: str, message: Dict[str, Any]) -> None:
        for attempt in range(self._max_retries):
            try:
                await self._attempt(topic, message)
                return
            except (IOError, httpx.RequestError) as exc:
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Invio fallito '{topic}' "
                        f"(tentativo {attempt+1}/{self._max_retries}). "
                        f"Retry in {wait}s. Errore: {type(exc).__name__}"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"Invio definitivamente fallito verso '{topic}' "
                        f"dopo {self._max_retries} tentativi."
                    )
                    raise RuntimeError(
                        f"Impossibile inviare l'evento a '{topic}'."
                    ) from exc
