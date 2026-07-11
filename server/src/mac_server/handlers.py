"""Message handlers for the MacBook server."""

from __future__ import annotations

import logging

from mac_server.protocol import make_ack_response
from shared.messages import Message


logger = logging.getLogger(__name__)


def handle_message(message: Message) -> Message:
    logger.info(
        "Handling message from device_id=%s type=%s payload=%s",
        message.device_id,
        message.type,
        message.payload,
    )
    return make_ack_response(message)
