# adapted from https://www.kaggle.com/code/huntingdata11/animated-and-interactive-nfl-plays-in-plotly
# Special thanks to Hunter Kempf for making his visualization code publicly available, which this is directly based on.
from argparse import ArgumentParser
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from animation.constants.field import FIELD_COLOR, SIDELINE_PADDING, X_HOME_ENDZONE, X_AWAY_GOAL_LINE, Y_BOTTOM_SIDELINE, Y_TOP_SIDELINE
from animation.constants.game import DOWNS
from animation.constants.subplots import SUBPLOT_BGCOLOR, SUBPLOT_DOMAIN_MAX, SUBPLOT_DOMAIN_MIN, SUBPLOT_Y_MIN, SUBPLOT_TITLES, SUBPLOT_TITLE_X, SUBPLOT_TITLE_Y
from animation.constants.teams import TEAM_COLORS, TEAM_NAMES
from animation.constants.ui import UPDATE_MENUS
from animation.draw_utils import (
    add_end_zone_text, add_first_down_markers, add_scoreline,
    draw_events, draw_horiz, draw_line_trace, draw_numbers, draw_vert,
    fill_end_zone,
    mark_events,
)
from animation.subplots import add_subplot_titles, static_graph_with_marker
from animation.text import get_title_str, next_play_description
from animation.ui_defaults import get_initial_slider_dict, get_slider_step
from datasets import PlayByPlayDataset
from preprocess_data import NFLBDBDataLoader

TACKLE_COLS = ["tackle", "assist", "pff_missedTackle"]

def get_layout(week, date, homeTeam, awayTeam, gameId, playId, playDescription, slider_dict, y0, y1, t, y, tmax=100, scale=20, multiplot=True):
    fig = make_subplots(
        rows=2,
        cols=4,
        row_heights=[0.7, 0.25],
        vertical_spacing=0.2,
        horizontal_spacing=0.2,
        specs=[
            [{"colspan": 4}, None, None, None],
            [{}, {}, {}, {}]
        ],)

    if multiplot:
        fig.update_layout(
            autosize=False,
            width=120 * scale,
            height=90 * scale,
            xaxis1=dict(domain=[0, 1], range=[0, 120], autorange=False, tickmode='array', tickvals=np.arange(10, 111, 5).tolist(), showticklabels=False),
            yaxis1=dict(domain=[0.32, 0.9], range=[Y_BOTTOM_SIDELINE - SIDELINE_PADDING, Y_TOP_SIDELINE + SIDELINE_PADDING], autorange=False, showgrid=False, showticklabels=False),
            xaxis2=dict(domain=[0, 0.2], range=[0, tmax], autorange=False),
            yaxis2=dict(domain=[SUBPLOT_DOMAIN_MIN, SUBPLOT_DOMAIN_MAX], range=[SUBPLOT_Y_MIN, 10], autorange=False),
            xaxis3=dict(domain=[0.26, 0.46], range=[0, tmax], autorange=False),
            yaxis3=dict(domain=[SUBPLOT_DOMAIN_MIN, SUBPLOT_DOMAIN_MAX], range=[SUBPLOT_Y_MIN, 15], autorange=False),
            xaxis4=dict(domain=[0.53, 0.73], range=[0, tmax], autorange=False),
            yaxis4=dict(domain=[SUBPLOT_DOMAIN_MIN, SUBPLOT_DOMAIN_MAX], range=[SUBPLOT_Y_MIN, 10], autorange=False),
            xaxis5=dict(domain=[0.8, 1.0], range=[0, tmax], autorange=False),
            yaxis5=dict(domain=[SUBPLOT_DOMAIN_MIN, SUBPLOT_DOMAIN_MAX], range=[SUBPLOT_Y_MIN, 10], autorange=False),
            plot_bgcolor=FIELD_COLOR,
            title=get_title_str(week, date, homeTeam, awayTeam, gameId, playId, playDescription, y0, y1, t, y),
            updatemenus=UPDATE_MENUS,
            sliders=[slider_dict]
        )

        fig['layout']['xaxis2']['title'] = 't'
        fig['layout']['xaxis3']['title'] = 't'
        fig['layout']['xaxis4']['title'] = 't'
        fig['layout']['xaxis5']['title'] = 't'
        fig['layout']['yaxis2']['title'] = 'yds'
        fig['layout']['yaxis3']['title'] = 'yds/s'
        fig['layout']['yaxis4']['title'] = 'yds/s'
        fig['layout']['yaxis5']['title'] = 'yds/s^2'
        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref=f"x{i}",
                    yref=f"y{i}",
                    x0=0,
                    y0=SUBPLOT_Y_MIN - 1,
                    x1=tmax,
                    y1=20,
                    fillcolor=SUBPLOT_BGCOLOR,
                    opacity=1.,
                    layer="below",
                    line_width=0,
                ) for i in range(2, 6)
            ]
        )
        return fig.layout
    else:
        return go.Layout(
            autosize=False,
            width=120 * scale,
            height=70 * scale,
            xaxis1=dict(range=[0, 120], autorange=False, tickmode='array', tickvals=np.arange(10, 111, 5).tolist(), showticklabels=False),
            yaxis1=dict(domain=[0, 0.9], range=[Y_BOTTOM_SIDELINE - SIDELINE_PADDING, Y_TOP_SIDELINE + SIDELINE_PADDING], autorange=False, showgrid=False, showticklabels=False),
            plot_bgcolor=FIELD_COLOR,
            title=get_title_str(week, date, homeTeam, awayTeam, gameId, playId, playDescription, y0, y1, t, y),
            updatemenus=UPDATE_MENUS,
            sliders=[slider_dict]
        )

def initialize_buffer(extra_keys=[]):
    keys = ["x", "y"] + extra_keys
    return {k: [] for k in keys}


def animate_play(tracking_df, game_df, play_df, tackle_df, players, gameId, playId, y0, y1, t, y, multiplot=False, animation_dir="../animations/"):
    anim_path = os.path.join(animation_dir, f'GAME{gameId}_PLAY{playId}.html')
    selected_game_df = game_df[game_df.gameId == gameId]
    selected_play_df = play_df[(play_df.playId == playId) & (play_df.gameId == gameId)]

    selected_tackle_df = tackle_df[(tackle_df.playId == playId) & (tackle_df.gameId == gameId)]
    tacklers = selected_tackle_df.nflId.tolist()

    tracking_players_df = pd.merge(tracking_df[(tracking_df.playId == playId) & (tracking_df.gameId == gameId)], players, how="left", on="nflId")
    selected_tracking_df = tracking_players_df[(tracking_players_df.playId == playId) & (tracking_players_df.gameId == gameId)]

    sorted_frame_list = selected_tracking_df.frameId.unique()
    sorted_frame_list.sort()
    play_direction_sign = (2 * (selected_tracking_df.iloc[0].playDirection == "right") - 1)
    # get play General information
    line_of_scrimmage = selected_play_df.absoluteYardlineNumber.values[0]
    first_down_marker = line_of_scrimmage + selected_play_df.yardsToGo.values[0] * play_direction_sign
    down = selected_play_df.down.values[0]
    quarter = selected_play_df.quarter.values[0]
    gameClock = selected_play_df.gameClock.values[0]
    yardsToGo = selected_play_df.yardsToGo.values[0]
    yardlineSide = selected_play_df.yardlineSide.values[0]
    yardlineNumber = selected_play_df.yardlineNumber.values[0]
    playDescription = selected_play_df.playDescription.values[0]
    possessionTeam = selected_play_df.possessionTeam.values[0]
    homeScore = selected_play_df.preSnapHomeScore.values[0]
    awayScore = selected_play_df.preSnapVisitorScore.values[0]
    homeTeam = selected_game_df.homeTeamAbbr.values[0]
    awayTeam = selected_game_df.visitorTeamAbbr.values[0]

    # Handle case where we have a really long Play Description and want to split it into two lines
    if len(playDescription.split(" ")) > 15 and len(playDescription) > 115:
        playDescription = " ".join(playDescription.split(" ")[0:16]) + "<br>" + " ".join(playDescription.split(" ")[16:])

    frames = []
    event_buffer = initialize_buffer(["event", "t"])
    football_buffer = initialize_buffer()
    ball_carrier_buffer = initialize_buffer()
    tackler_buffer = {tackler_id: initialize_buffer() for tackler_id in tacklers}
    slider_dict = get_initial_slider_dict()
    seen_first_contact = False
    first_contact_x = 0.

    for frameId in tqdm(sorted_frame_list, desc="Generating animation"):
        data = []
        data.append(draw_numbers(Y_BOTTOM_SIDELINE + 5))
        data.append(draw_numbers(Y_TOP_SIDELINE - 5))
        data.append(draw_vert(line_of_scrimmage, 'blue', line_dash="solid", opacity=0.7))
        data.append(draw_vert(first_down_marker, 'yellow', line_dash="solid", opacity=0.7))
        data.append(draw_horiz(Y_TOP_SIDELINE, 'white'))

        data += fill_end_zone(homeTeam, awayTeam, possessionTeam, play_direction_sign)

        # Plot Players
        ball_carrier_df = selected_tracking_df.loc[selected_tracking_df.nflId == selected_play_df.ballCarrierId.item()]
        for club in selected_tracking_df.club.unique():
            plot_df = selected_tracking_df[(selected_tracking_df.club == club) & (selected_tracking_df.frameId == frameId)]
            if club != "football":
                hover_text_array = []
                for nflId in plot_df.nflId:
                    selected_player_df = plot_df[plot_df.nflId == nflId]
                    hover_str = ""
                    playerinfo = f"{selected_player_df['displayName'].values[0]} ({ selected_player_df['position'].values[0]})<br>"
                    playertrack = f"<i>(s: {selected_player_df['s'].item():.1f}, a: {selected_player_df['a'].item():.1f}, o: {selected_player_df['o'].item():.1f}, dir: {selected_player_df['dir'].item():.1f})</i><br>"
                    if nflId == selected_play_df.ballCarrierId.item():
                        action_str = "<b>*BALL CARRIER*</b><br>"
                        hover_str += action_str
                        if frameId % 3 == 0:
                            ball_carrier_buffer["x"].append(selected_player_df["x"].item())
                            ball_carrier_buffer["y"].append(selected_player_df["y"].item())
                        if multiplot:
                            data += static_graph_with_marker(ball_carrier_df, "s", frameId, action_str + playerinfo, TEAM_COLORS[club], "x4", "y4")
                            data += static_graph_with_marker(ball_carrier_df, "a", frameId, action_str + playerinfo, TEAM_COLORS[club], "x5", "y5")
                        data.append(draw_line_trace(ball_carrier_buffer, color=TEAM_COLORS[club], opacity=0.9, size=2, line_dash='solid'))

                    elif nflId in tacklers:
                        role = selected_tackle_df.loc[selected_tackle_df.nflId == nflId, TACKLE_COLS].idxmax(axis=1).item().replace("pff_missedTackle", "missed tackle")
                        action_str = f"<b>*ACTION: {role}*</b><br>"
                        hover_str += action_str
                        tackler_buffer[nflId]["x"].append(plot_df.loc[plot_df.nflId == nflId, "x"].item())
                        tackler_buffer[nflId]["y"].append(plot_df.loc[plot_df.nflId == nflId, "y"].item())
                        data.append(draw_line_trace(tackler_buffer[nflId], color=TEAM_COLORS[club], opacity=0.8, size=2, line_dash='solid'))

                        if multiplot:
                            tackler_df = selected_tracking_df.loc[selected_tracking_df.nflId == nflId]
                            # compute distance to ball carrier and angle similarity
                            tackler_df = tackler_df.assign(
                                distance=np.linalg.norm(tackler_df[["x", "y"]].values - ball_carrier_df[["x", "y"]].values, axis=1),
                                relvel=np.sqrt(tackler_df["s"].values ** 2 + ball_carrier_df["s"].values ** 2 - 2 * tackler_df["s"].values * ball_carrier_df["s"].values * np.cos((tackler_df["o"].values - ball_carrier_df["o"].values) * np.pi / 180))
                            )
                            data += static_graph_with_marker(tackler_df, "distance", frameId, action_str + playerinfo, TEAM_COLORS[club], "x2", "y2")
                            data += static_graph_with_marker(tackler_df, "relvel", frameId, action_str + playerinfo, TEAM_COLORS[club], "x3", "y3")

                    hover_str += playerinfo
                    hover_str += playertrack
                    hover_text_array.append(hover_str)

                data.append(go.Scatter(
                    x=plot_df["x"], y=plot_df["y"],
                    mode='markers',
                    marker=dict(
                        color=TEAM_COLORS[club],
                        line=dict(width=1, color="white"),
                        size=10,
                        symbol="circle" if club == selected_play_df.possessionTeam.values[0] else "square",
                    ),
                    name=club,
                    hovertext=hover_text_array,
                    hoverinfo="text"))
            else:
                event = plot_df.event.item()
                if frameId % 3 == 0:
                    football_buffer["x"].append(plot_df["x"].item())
                    football_buffer["y"].append(plot_df["y"].item())
                if not pd.isnull(event):
                    if event == "first_contact":
                        seen_first_contact = True
                        first_contact_x = plot_df["x"].item()
                    event_buffer["x"].append(plot_df["x"].item())
                    event_buffer["y"].append(plot_df["y"].item())
                    event_buffer["event"].append(event)
                    event_buffer["t"].append(frameId)
                if seen_first_contact:
                    y0_text = f"Projected line of scrimmage, tackle unsuccessful:<br><i>{y0:.2f} yds after contact<i><br>"
                    y1_text = f"Projected line of scrimmage, tackle successful:<br><i>{y1:.2f} yds after contact<i><br>"
                    y0_absolute = first_contact_x + y0 * play_direction_sign
                    y1_absolute = first_contact_x + y1 * play_direction_sign
                    y0_text += next_play_description(y0_absolute, down, yardsToGo, line_of_scrimmage, first_down_marker, homeTeam, awayTeam, possessionTeam)
                    y1_text += next_play_description(y1_absolute, down, yardsToGo, line_of_scrimmage, first_down_marker, homeTeam, awayTeam, possessionTeam)

                    data.append(draw_vert(y0_absolute, "#00ff00", line_dash="dot", text=y0_text))
                    data.append(draw_vert(y1_absolute, "red", line_dash="dot", text=y1_text))
                else:
                    data.append(draw_vert(-100, "red", opacity=0, text=""))
                    data.append(draw_vert(-100, "red", opacity=0, text=""))

                data.append(draw_events(event_buffer))
                data.append(draw_line_trace(football_buffer))
                data.append(go.Scatter(
                    x=plot_df["x"],
                    y=plot_df["y"],
                    mode='markers',
                    marker_color=TEAM_COLORS[club],
                    name=club,
                    hoverinfo='none',
                    marker_symbol="diamond-wide")
                )

        # add frame to slider
        slider_step = get_slider_step(frameId, (event_buffer["event"][-1] + f' {event_buffer["t"][-1]}') if len(event_buffer["event"]) else "N/A")
        slider_dict["steps"].append(slider_step)
        frames.append(go.Frame(data=data, name=str(frameId)))

    week = selected_game_df.week.values[0]
    date = selected_game_df.gameDate.values[0]
    layout = get_layout(week, date, homeTeam, awayTeam, gameId, playId, playDescription, slider_dict, y0, y1, t, y, tmax=len(frames), scale=10, multiplot=multiplot)
    fig = go.Figure(data=frames[0]["data"], layout=layout, frames=frames[1:])
    # Create First Down Markers
    add_first_down_markers(fig, line_of_scrimmage, down)
    add_end_zone_text(fig, homeTeam, awayTeam, possessionTeam, play_direction_sign)
    add_scoreline(
        fig,
        gameClock, quarter, down,
        yardsToGo, yardlineSide, yardlineNumber,
        possessionTeam,
        homeScore, awayScore,
        homeTeam, awayTeam
    )
    if multiplot:
        add_subplot_titles(fig)
    fig.write_html(anim_path)
    print("Saved animation to", anim_path)
    return fig

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--ckpt-dir", type=str, required=True)
    psr.add_argument("--indices", nargs="+", type=int, required=True)
    psr.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    psr.add_argument("--animation-dir", type=str, default="animation/animations/")
    psr.add_argument("--multiplot", action="store_true")
    args = psr.parse_args()

    raw_data = NFLBDBDataLoader()  # load data
    path = os.path.join(raw_data.base_path, f"play_by_play_{args.split}.pkl")
    dataset = PlayByPlayDataset(path)

    model_results = pickle.load(open(f"{args.ckpt_dir}/results.pkl", "rb"))

    for n_play, i in enumerate(args.indices):
        print(f"Animating play {args.split}:{i} ({n_play + 1}/{len(args.indices)})")
        if i >= len(dataset):
            print("Index out of bounds:", i, f"( max: {len(dataset) - 1})")
            continue
        animate_play(
            raw_data.tracking_data,
            raw_data.games,
            raw_data.plays,
            raw_data.tackles,
            raw_data.players,
            dataset[i]["game_id"],
            dataset[i]["play_id"],
            model_results["y0_pred"][i],
            model_results["y1_pred"][i],
            model_results["t_true"][i],
            model_results["y_true"][i],
            animation_dir=args.animation_dir,
            multiplot=args.multiplot,
        )
