def get_initial_slider_dict():
    return {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 30, "l": 30},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "currentvalue": dict(prefix="t="),
        "steps": []
    }

def get_slider_step(frameId, label):
    return {
        "args": [
            [frameId],
            {"frame": {"duration": 100, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 0}}
        ],
        "label": str(frameId),
        "method": "animate"
    }
