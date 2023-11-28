import numpy as np
import plotly.graph_objects as go

from animation.constants.field import (
    X_HOME_ENDZONE,
    X_AWAY_ENDZONE,
    X_HOME_10YD,
    X_AWAY_GOAL_LINE,
    X_NUMBER_SPACING,
    X_N_NUMBERS,
    Y_BOTTOM_SIDELINE,
    Y_TOP_SIDELINE,
    YARD_NUMBERS,
)
from animation.constants.game import DOWNS
from animation.constants.teams import TEAM_COLORS, TEAM_NAMES
from animation.constants.scoreline import SCORELINE_Y, SCORELINE_Y_SINGLEPLOT
from animation.text import conditional_bold, possession_indicator

def draw_vert(x, color, line_dash='dash', text=None, opacity=1., resolution=5, xaxis="x1", yaxis="y1"):
    if text is None:
        x = [x, x]
        y = [Y_BOTTOM_SIDELINE, Y_TOP_SIDELINE]
    else:
        x = [x] * resolution
        y = np.linspace(Y_BOTTOM_SIDELINE, Y_TOP_SIDELINE, resolution)
    return go.Scatter(
        x=x,
        y=y,  # [0, 53.5],
        mode="lines+text",
        line_dash=line_dash,
        line_color=color,
        showlegend=False,
        opacity=opacity,
        hovertext=text,
        hoverinfo="text",
        xaxis=xaxis,
        yaxis=yaxis,
    )

def draw_horiz(y, color, line_dash='solid', text=None, opacity=1.):
    return go.Scatter(
        x=[X_HOME_ENDZONE, X_AWAY_ENDZONE],
        y=[y, y],
        mode="lines+text",
        line_dash=line_dash,
        line_color=color,
        showlegend=False,
        opacity=opacity,
        hovertext=text,
        hoverinfo="text",
    )

def draw_numbers(y):
    return go.Scatter(
        x=np.arange(X_HOME_10YD, X_AWAY_GOAL_LINE, X_NUMBER_SPACING),
        y=[y] * X_N_NUMBERS,
        mode='text',
        text=YARD_NUMBERS,
        textfont_size=30,
        textfont_family="Courier New, monospace",
        textfont_color="white",
        showlegend=False,
        hoverinfo='none'
    )

def add_first_down_markers(fig, x, down):
    fig.add_annotation(
        x=x,
        y=Y_TOP_SIDELINE - 0.5,
        text=str(down),
        showarrow=False,
        font=dict(family="Courier New, monospace", size=16, color="black"),
        align="center",
        bordercolor="black",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=1
    )

def add_scoreline(
    fig,
    gameClock,
    quarter,
    down,
    yardsToGo,
    yardlineSide,
    yardlineNumber,
    possessionTeam,
    homeScore,
    awayScore,
    homeTeam,
    awayTeam,
    multiplot=False,
):
    # SCORELINE_KWARGS = {
    #    "y": SCORELINE_Y,
    #    "yref": "paper",
    #    "showarrow": False,
    #    "font": {"family": "Courier New, monospace", "size": 34, "color": "white"},
    #    "opacity": 1.,
    #    "bordercolor": "black",
    #    "borderwidth": 2,
    #    "borderpad": 2,
    # }
    gamefont = {"family": "Courier New, monospace", "size": 34, "color": "white"}
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=SCORELINE_Y if multiplot else SCORELINE_Y_SINGLEPLOT, text=conditional_bold((" " + homeTeam + possession_indicator(homeTeam, possessionTeam)).ljust(18)), showarrow=False, font=gamefont,
        align="left", bgcolor=TEAM_COLORS[homeTeam], opacity=1, bordercolor="black", borderwidth=2, borderpad=2)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.975, y=SCORELINE_Y if multiplot else SCORELINE_Y_SINGLEPLOT, text=conditional_bold((" " + awayTeam + possession_indicator(awayTeam, possessionTeam)).ljust(18)), showarrow=False, font=gamefont,
        align="left", bgcolor=TEAM_COLORS[awayTeam], opacity=1, bordercolor="black", borderwidth=2, borderpad=2)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.355, y=SCORELINE_Y if multiplot else SCORELINE_Y_SINGLEPLOT, text=conditional_bold(" " + str(homeScore).ljust(3)), showarrow=False, font=gamefont,
        align="center", bgcolor=adjust_color(TEAM_COLORS[homeTeam]), opacity=1, bordercolor="black", borderwidth=2, borderpad=2)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1., y=SCORELINE_Y if multiplot else SCORELINE_Y_SINGLEPLOT, text=conditional_bold(" " + str(awayScore).ljust(3)), showarrow=False, font=gamefont,
        align="center", bgcolor=adjust_color(TEAM_COLORS[awayTeam]), opacity=1, bordercolor="black", borderwidth=2, borderpad=2)

    if yardlineNumber == 50:
        yardline_str = "50"
    else:
        yardline_str = f"{yardlineSide} {yardlineNumber}"
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=SCORELINE_Y if multiplot else SCORELINE_Y_SINGLEPLOT, text=conditional_bold(f"  Q{quarter} {gameClock}  <br>  {DOWNS[down-1]} & {yardsToGo} | Ball on {yardline_str}  "), showarrow=False,
        align="center", bgcolor="black", opacity=1, font={"family": "Helvetica, sans-serif", "size": 16, "color": "white"},
        bordercolor="black", borderwidth=2, borderpad=2,
    )

def draw_events(event_buffer):
    return go.Scatter(
        x=event_buffer["x"], y=event_buffer["y"],
        mode="markers+text",
        marker=dict(symbol="x", color="white"),
        hoverinfo="text",
        hovertext=[f't={event_t}: {events}' for event_t, events in zip(event_buffer['t'], event_buffer["event"])],
        textposition="top left",
        textfont=dict(family="Courier New, monospace", size=14, color="white"),
        text=event_buffer["event"], marker_symbol="x",
        marker_color="white", showlegend=False,
    )

def mark_events(event_df, xaxis, yaxis):  # unused
    return [
        go.Scatter(
            x=[event_t, event_t, event_t, event_t],
            y=[-1, 0, 5, 10],
            mode="lines",
            line=dict(color="red", dash="dot", width=1.),
            hoverinfo="text",
            hovertext=f't={event_t}: {event}',
            showlegend=False, xaxis=xaxis, yaxis=yaxis
        ) for event_t, event in zip(event_df.frameId, event_df.event)
    ]

def draw_line_trace(buffer, color='white', opacity=1., line_dash='dot', size=2.):
    return go.Scatter(
        x=buffer["x"],
        y=buffer["y"],
        mode='lines',
        line=dict(color=color, dash=line_dash, width=size),
        hoverinfo='none',
        opacity=opacity,
        showlegend=False,
    )


def fill_end_zone(home, away, poss_team, direction):
    data = []
    non_poss_team = home if away == poss_team else away
    left_team = poss_team if direction > 0 else non_poss_team  # if moving right, then possession team's end zone is on the left
    right_team = poss_team if direction < 0 else non_poss_team
    for x_min in [X_HOME_ENDZONE, X_AWAY_GOAL_LINE]:
        data.append(
            go.Scatter(
                x=[x_min, x_min, x_min + 10, x_min + 10, x_min],
                y=[Y_BOTTOM_SIDELINE, Y_TOP_SIDELINE, Y_TOP_SIDELINE, Y_BOTTOM_SIDELINE, Y_BOTTOM_SIDELINE],
                fill="toself",
                fillcolor=TEAM_COLORS[left_team] if x_min == X_HOME_ENDZONE else TEAM_COLORS[right_team],
                mode="lines",
                line=dict(
                    color="white",
                    width=3
                ),
                opacity=1,
                showlegend=False,
                hoverinfo="skip"
            )
        )
    return data

def add_end_zone_text(fig, home, away, poss_team, direction):
    non_poss_team = home if away == poss_team else away
    left_team = poss_team if direction > 0 else non_poss_team  # if moving right, then possession team's end zone is on the left
    right_team = poss_team if direction < 0 else non_poss_team
    for x_min in [X_HOME_ENDZONE, X_AWAY_GOAL_LINE]:
        fig.add_annotation(
            x=x_min + 5,
            y=Y_TOP_SIDELINE / 2,
            text=f"<b>{TEAM_NAMES[left_team if x_min == X_HOME_ENDZONE else right_team]}</b>",
            showarrow=False,
            font=dict(
                family="Helvetica, sans-serif",
                size=32,
                color="white",
            ),
            textangle=90 if x_min == X_HOME_ENDZONE else 270
        )

def darken_color(r, g, b, factor=0.6, flip=False):  # This function was generated by ChatGPT and double-checked manually.
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    r = max(0, min(r, 255))
    g = max(0, min(g, 255))
    b = max(0, min(b, 255))
    if flip:
        r = 255 - r
        g = 255 - g
        b = 255 - b
    # Convert RGB back to hex
    darkened_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return darkened_hex


def adjust_color(hex_color):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    if (r + g + b) > 110:  # too bright; darken -- pretty conservative threshold here
        return darken_color(r, g, b)
    else:
        return darken_color(255 - r, 255 - g, 255 - b, flip=True)
