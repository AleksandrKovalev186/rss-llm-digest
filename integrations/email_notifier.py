from integrations.message_state import State

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from settings.config import settings


async def email_node(state: State) -> State:
    _from = settings.email_from
    _to = settings.email_to
    msg = MIMEMultipart()
    msg["From"] = _from
    msg["To"] = _to
    msg["Subject"] = 'RSS NEWS FROM AI AGENT'
    msg.attach(MIMEText(state.summaries, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(_from, settings.email_password)
        server.sendmail(_from, _to, msg.as_string())

    return state
