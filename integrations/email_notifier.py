from integrations.message_state import State


async def email_node(state: State) -> State:
    print("Email node: not implemented yet")
    state.email_sent = False
    return state
