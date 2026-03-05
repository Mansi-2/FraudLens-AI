def simulate_transaction(base_input, amount=None, time=None):
    simulated = base_input.copy()

    if amount is not None:
        simulated["Amount"] = amount

    if time is not None:
        simulated["Time"] = time

    return simulated