"""Send the email verification code that a booking needs.

Production sends through an SMTP server, for example a Gmail account with an app password
(free, about 500 emails a day). Without SMTP settings, local runs print the code to the log
instead; production refuses to start that way.
"""

import asyncio
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Protocol

import structlog

from ragmed_core.clinic import CLINIC_NAME

log = structlog.get_logger()

SUBJECT = f"Your {CLINIC_NAME} verification code"


def code_email(code: str, minutes: int) -> str:
    return (
        f"Your verification code for {CLINIC_NAME} is:\n\n"
        f"    {code}\n\n"
        f"Enter it in the chat to confirm your visit. It expires in {minutes} minutes.\n\n"
        "If you did not ask to book a visit, please ignore this message: nothing will be booked "
        "without this code.\n\n"
        f"{CLINIC_NAME} is an educational project, not a medical service."
    )


class MailError(Exception):
    """The email could not be sent; the message is safe to show the patient."""


class Mailer(Protocol):
    async def send_code(self, to: str, code: str, minutes: int) -> None: ...


@dataclass(frozen=True, slots=True)
class SmtpMailer:
    host: str
    port: int
    username: str
    password: str
    sender: str
    timeout: float = 15

    def _send(self, message: EmailMessage) -> None:
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as smtp:
            smtp.starttls()
            smtp.login(self.username, self.password)
            smtp.send_message(message)

    async def send_code(self, to: str, code: str, minutes: int) -> None:
        message = EmailMessage()
        message["Subject"] = SUBJECT
        message["From"] = f"{CLINIC_NAME} <{self.sender}>"
        message["To"] = to
        message.set_content(code_email(code, minutes))
        try:
            await asyncio.to_thread(self._send, message)
        except smtplib.SMTPRecipientsRefused as error:
            raise MailError("That email address was refused. Please check it.") from error
        except (smtplib.SMTPException, OSError) as error:
            log.error("verification email failed", error=str(error))
            raise MailError("We could not send the email just now. Please try again.") from error


class ConsoleMailer:
    """Local development only: writes the code to the log instead of sending it."""

    async def send_code(self, to: str, code: str, minutes: int) -> None:  # noqa: ARG002
        log.warning("verification code (local only, not emailed)", to=to, code=code)


@dataclass
class RecordingMailer:
    """For tests: keeps every code it was asked to send."""

    sent: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None

    async def send_code(self, to: str, code: str, minutes: int) -> None:  # noqa: ARG002
        if self.error:
            raise MailError(self.error)
        self.sent.append((to, code))
