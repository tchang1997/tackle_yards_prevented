UPDATE_MENUS = [
    {
        "buttons": [
            {
                "args": [
                    None,
                    {
                        "frame": {"duration": 100, "redraw": False},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }
                ],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ],
                "label": "Pause",
                "method": "animate"
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 47},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
]
