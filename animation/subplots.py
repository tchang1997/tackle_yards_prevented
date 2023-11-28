import plotly.graph_objects as go

from animation.constants.subplots import SUBPLOT_TITLE_X, SUBPLOT_TITLE_Y, SUBPLOT_TITLES
from animation.draw_utils import adjust_color

def add_subplot_titles(fig):
    for i in range(4):
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=SUBPLOT_TITLE_X[i],
            y=SUBPLOT_TITLE_Y,
            text=SUBPLOT_TITLES[i],
            align="center",
            showarrow=False,
            font={"family": "Helvetica, sans-serif", "size": 16}
        )

def static_graph_with_marker(df, col, frameId, extra_text, color, xaxis, yaxis):
    val = df.loc[df.frameId == frameId, col].item()
    return [
        go.Scatter(
            y=df[col],
            mode='lines',
            line=dict(color=color, dash='solid', width=2),
            xaxis=xaxis,
            yaxis=yaxis,
            opacity=0.7,
            showlegend=False,
            hoverinfo="text"
        ),
        go.Scatter(
            x=[frameId],
            y=[val],
            mode='markers',
            marker=dict(
                color=adjust_color(color),
                size=8,
                symbol="circle",
                opacity=1.,
            ),
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=False,
            hoverinfo="text",
            hovertext=f"t={frameId}: {val:.2f}<br>" + extra_text,),
    ]
