from yaspin import yaspin

def get_default_text_spinner_context(text, color="green", timer=True):
    return yaspin(text=text, color=color, timer=timer)
