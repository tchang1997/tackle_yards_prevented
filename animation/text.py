import numpy as np

from animation.constants.game import DOWNS

def conditional_bold(str, passthrough=False):
    if passthrough:
        return str
    return f"<b>{str}</b>"

def get_title_str(week, date, homeTeam, awayTeam, gameId, playId, playDescription, y0, y1, t, y):
    factual_error = y - y0 if t == 0 else y - y1
    status_str = "prevented" if t == 1 else "preventable"
    sign = "more than" if factual_error > 0 else "less than"
    return f"<b>Week {week} ({date}): {homeTeam} vs. {awayTeam}</b> [game_id: {gameId}, play_id: {playId}]<br>{playDescription}<br>" + \
        f"<b>Expected yards {status_str}: {y0-y1:.1f}yds</b> " + \
        "(" + conditional_bold(f"if tackle successful: {y1:.1f}yds", passthrough=(t == 0)) + ", " + conditional_bold(f"if tackle failed: {y0:.1f}yds", passthrough=(t == 1)) + ")<br>" + \
        f"Actual: {y:.1f}yds (ball carrier gained {abs(factual_error):.1f}yds {sign} expected)<br>"

def possession_indicator(team, possessionTeam):
    return " <>" if team == possessionTeam else ""

def get_left_right_team(home, away, poss_team, direction):
    non_poss_team = home if away == poss_team else away
    left_team = poss_team if direction > 0 else non_poss_team  # if moving right, then possession team's end zone is on the left
    right_team = poss_team if direction < 0 else non_poss_team
    return left_team, right_team

def next_play_description(new_yardline, down, yardsToGo, line_of_scrimmage, first_down, homeTeam, visitorTeam, possessionTeam):
    play_direction = np.sign(first_down - line_of_scrimmage)  # positive = right, negative = left
    nextToGo = int((first_down - new_yardline) * play_direction)
    next_first_down = first_down
    left_team, right_team = get_left_right_team(homeTeam, visitorTeam, possessionTeam, play_direction)
    play_text = ""

    if nextToGo <= 0:
        next_down = 1
        nextToGo = 10
        next_first_down = first_down + 10 * play_direction
    else:
        next_down = down + 1

    if next_first_down > 110 or next_first_down < 10:
        nextToGo = "G"

    yardline_str = ""
    if new_yardline > 60:
        yardline_str = f"{right_team} {int(110 - new_yardline)}"
    elif new_yardline < 60:
        yardline_str = f"{left_team} {int(new_yardline - 10)}"
    else:
        yardline_str = "50"

    if new_yardline < 10:
        if play_direction < 0:
            play_text = f"{right_team} TD"
        else:
            play_text = f"{right_team} Safety"
    elif new_yardline > 110:
        if play_direction > 0:
            play_text = f"{left_team} TD"
        else:
            play_text = f"{left_team} Safety"
    elif next_down <= 4:
        play_text = f"{DOWNS[next_down - 1]} & {nextToGo} at {yardline_str}"
    else:
        play_text = f"Turnover on downs at {yardline_str}"
    return play_text
