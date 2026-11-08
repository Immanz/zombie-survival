"""
🧟 Zombie Survival — a choose-your-own-adventure game powered by Gemini
Run with:  streamlit run app.py
"""

import os
import random
import time
from pathlib import Path
from typing import Literal

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel


# =====================================================================
# 1. SETTINGS  — the easiest things to change are all up here
# =====================================================================

GAME_TITLE = "🧟 Zombie Survival"
CHOICES_TO_SURVIVE = 10         # make this many choices without dying = you survive

# Models to try, in order. The "lite" models are the fastest, so they go first.
# If one is busy (503), too slow, or missing (404), the game tries the next one.
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]
SECONDS_PER_TRY = 20   # give up on a model after this many seconds and try the next

# End-screen clips. Put your files in a folder called "assets" next to app.py.
# They can be .gif or .mp4. If a file is missing, a built-in drawing is shown instead.
ASSETS_FOLDER = Path(__file__).parent / "assets"
LOSE_CLIP = "lose.gif"    # shown when the player dies
WIN_CLIP = "win.gif"      # shown when the player survives


# ---------------------------------------------------------------------
# YOUR STORY SETTINGS  —  change these to control the story.
# Single values ("text") are used every game.
# Lists ([...]) -> the game randomly picks ONE item each new game.
# ---------------------------------------------------------------------

STORY_SETTINGS = {
    # --- the same every game ---
    "city": "Buffalo, New York",
    "goal": "reach the military rescue helicopter waiting on the roof of a hospital downtown",
    "zombie_type": "slow, but they travel in huge hordes and are attracted to noise",
    "tone": "tense and scary, with a little dark humor",
    "difficulty": "hard",   # pick one: "easy", "medium", "hard"

    # --- randomly picked each game ---
    "starting_location": [
        "a lecture hall on UB North Campus",
        "a Tim Hortons on Main Street",
        "a stuck subway car under downtown",
        "your bedroom at home, at 3 a.m.",
        "the Walden Galleria mall",
    ],
    "time_of_day": ["early morning", "afternoon", "sunset", "the middle of the night"],
    "weather": ["a heavy snowstorm", "pouring rain", "thick fog", "a clear, quiet night"],
    "companion": [
        "nobody, you are alone",
        "your best friend, who is very scared",
        "a loyal dog that follows you",
        "a stranger who seems to be hiding something",
    ],
    "secret_twist": [
        "the player's companion has been bitten and is hiding it",
        "the helicopter will leave earlier than expected",
        "a group of other survivors is NOT friendly",
        "the zombies are getting faster as the story goes on",
        "there is no twist this time",
    ],
}

# How much health the "hurt" choice takes away, for each difficulty (min, max)
DAMAGE = {
    "easy":   (10, 20),
    "medium": (15, 30),
    "hard":   (25, 40),
}


# =====================================================================
# 2. THE STORY RULES GEMINI FOLLOWS
# =====================================================================

def pick_this_game():
    """Lock in the settings for ONE game. Lists get a random pick."""
    game = {}
    for key, value in STORY_SETTINGS.items():
        game[key] = random.choice(value) if isinstance(value, list) else value
    return game


def build_prompt(game):
    """Turn this game's settings into the rules Gemini follows."""
    return f"""
You are the narrator of a zombie survival choose-your-own-adventure game.
Invent a fresh, original story every time.

STORY SETTINGS FOR THIS GAME:
- City: {game["city"]}
- The player starts in: {game["starting_location"]}
- Time of day: {game["time_of_day"]}
- Weather: {game["weather"]}
- Companion: {game["companion"]}
- Zombies: {game["zombie_type"]}
- Goal: {game["goal"]}
- Tone: {game["tone"]}
- SECRET TWIST (never say it directly; hint at it, then reveal it in the
  second half of the story): {game["secret_twist"]}

WRITING STYLE (very important):
- 2 to 3 sentences per scene. Never more than 50 words.
- Short, punchy sentences. Second person ("You...").
- Include one vivid detail (a sound, a smell, something moving).
- End every scene on a hook or cliffhanger.
- No filler, no repeating what already happened, no graphic gore.

CHOICES:
- Give exactly 3 choices. Each is under 10 words and starts with a verb.
- Tag each choice with a hidden outcome. You MUST use each one exactly once:
    "death" -> this choice kills the player
    "hurt"  -> this choice injures the player
    "safe"  -> nothing bad happens
- All 3 choices must sound reasonable. Never make the deadly one obvious,
  but leave a small clue in the scene that a careful player could notice.

WHEN THE PLAYER PICKS:
- You will be told the outcome of their pick. Start the next scene by
  showing that consequence, then continue the story.
- If told the player DIED: write a short, dramatic death scene (max 50 words)
  and give an empty list of choices.
- If told the player SURVIVED: write a short, exciting escape ending (max 50 words)
  and give an empty list of choices.
- The game lasts {CHOICES_TO_SURVIVE} choices. Slowly move toward the goal.
- VERY IMPORTANT: the player must NOT reach the goal, board the helicopter,
  or escape until you are told they SURVIVED. Until then, always add a new
  obstacle and give 3 new choices. Never write "you survived" early.
"""


# =====================================================================
# 3. THE SHAPE OF GEMINI'S ANSWER (structured output)
# =====================================================================

class Choice(BaseModel):
    text: str
    outcome: Literal["death", "hurt", "safe"]


class Turn(BaseModel):
    story: str
    choices: list[Choice]


# =====================================================================
# 4. TALKING TO GEMINI
# =====================================================================

def get_api_key():
    """Use the key from Streamlit secrets (when deployed online)
    or from the GEMINI_API_KEY environment variable (on your laptop)."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


client = genai.Client(
    api_key=get_api_key(),
    http_options=types.HttpOptions(
        timeout=SECONDS_PER_TRY * 1000,                     # give up on a slow call
        retry_options=types.HttpRetryOptions(attempts=1),   # we do our own retries below
    ),
)


def ask_gemini_once(model, history, low_thinking=True):
    """Ask ONE model for the next Turn."""
    config = types.GenerateContentConfig(
        system_instruction=st.session_state.prompt,
        response_mime_type="application/json",
        response_schema=Turn,
        temperature=1.0,
    )
    if low_thinking:
        config.thinking_config = types.ThinkingConfig(thinking_level="LOW")  # faster answers

    response = client.models.generate_content(model=model, contents=history, config=config)
    if response.parsed is not None:
        return response.parsed
    if response.text:
        return Turn.model_validate_json(response.text)
    raise RuntimeError("Gemini sent back an empty answer.")


def ask_gemini(history):
    """Try each model in MODELS until one works. Prints timing in the terminal."""
    last_error = None
    for model in MODELS:
        start = time.time()
        try:
            try:
                turn = ask_gemini_once(model, history)
            except Exception as error:
                # Some models don't support the thinking setting -> try once without it
                if "400" in str(error) and "think" in str(error).lower():
                    turn = ask_gemini_once(model, history, low_thinking=False)
                else:
                    raise
            print(f"✅ [{model}] answered in {time.time() - start:.1f} seconds")
            return turn
        except Exception as error:
            last_error = error
            print(f"❌ [{model}] failed after {time.time() - start:.1f} seconds: {str(error)[:120]}")
    raise last_error


def add_message(role, text):
    """Save one message ('user' = player, 'model' = Gemini) to the history."""
    st.session_state.history.append(
        types.Content(role=role, parts=[types.Part(text=text)])
    )


def fix_choices(turn):
    """Make sure there is exactly one death, one hurt and one safe choice,
    then shuffle them so the deadly one isn't always in the same spot."""
    if not turn.choices:
        return
    outcomes = sorted(c.outcome for c in turn.choices)
    if len(turn.choices) != 3 or outcomes != ["death", "hurt", "safe"]:
        turn.choices = turn.choices[:3]
        while len(turn.choices) < 3:
            turn.choices.append(Choice(text="Run for it", outcome="safe"))
        for choice, outcome in zip(turn.choices, random.sample(["death", "hurt", "safe"], 3)):
            choice.outcome = outcome
    random.shuffle(turn.choices)


def take_turn(player_message, ending=None):
    """Player does something -> Gemini answers -> save everything.
    ending is None (game goes on), 'dead', or 'survived'."""
    add_message("user", player_message)
    try:
        with st.spinner("Something is moving in the dark..."):
            turn = ask_gemini(st.session_state.history)
    except Exception as error:
        st.session_state.history.pop()          # undo so the player can try again
        st.session_state.error = str(error)     # show the error after the page reloads
        return False

    if ending:
        turn.choices = []                       # the game is over: no more choices
        st.session_state.ending = ending
    else:
        if len(turn.choices) < 3:
            # Gemini forgot the choices (or ended the story early) -> ask once more
            try:
                retry_history = st.session_state.history + [
                    types.Content(role="model", parts=[types.Part(text=turn.model_dump_json())]),
                    types.Content(role="user", parts=[types.Part(text=(
                        "The game is NOT over. The player has not escaped yet. "
                        "Rewrite that scene with a new obstacle and give exactly 3 choices."))]),
                ]
                with st.spinner("The story takes a turn..."):
                    turn = ask_gemini(retry_history)
            except Exception:
                pass
        if len(turn.choices) < 3:
            # Still nothing -> backup choices so the game never gets stuck
            turn.choices = [
                Choice(text="Hide and wait", outcome="safe"),
                Choice(text="Push forward fast", outcome="hurt"),
                Choice(text="Open the nearest door", outcome="death"),
            ]
        fix_choices(turn)

    add_message("model", turn.model_dump_json())
    st.session_state.turns.append(turn)
    return True

game_story = {
    "turn 1": {
        "story": " You are at home on the couch, eating chips and buttered popcorn while watching a TV show. However, you get an alert on your phone with the message “Warning! A catastrophic event is unfolding. A zombie outbreak has occurred globally, and more than 40 million people have been infected by an unknown fungal virus.” You glanced at your phone with such concern and confusion that you turned on the news outlet; the reporters' faces were pale because on the screen there was blood splattered on the camera with a zombie biting a person's neck. You get up from the couch, tossing the popcorn and chip bag to the side, forcing you to think fast.",
        "choices": ["Go to your room and get items you would desperately need to survive", "Lock yourself inside the bathroom", "Disregard the alert and go to sleep because you think it’s all a dream"]
    }, 
}
    "turn_2_A": {
        "story": "You scramble to your room, pick up a backpack lying on the floor, and dump clothes and water bottles into it to stay hydrated and fresh throughout your journey. You also grab a couple of other snacks from the counter drawer. When you check your cellphone, you find out that the nearest shelter is an 11-hour walk away. You sit there, thinking and analyzing whether it's truly possible to get there with so many people getting infected every hour. Then suddenly, you hear a sharp tapping sound echoing directly against your bedroom window. What do you do in this situation?",
        "choices": ["Run to the roof", "Hide in the closet", "Call for help"]
    },
    "turn_2_B": {
        "story": "You lock yourself inside the bathroom, hoping that the zombies won't find you. You hear scratching and banging on the door, and you realize that you need to find a way out. You look around the bathroom and see a small window that leads to the roof. What do you do?",
        "choices": ["Stay in the bathroom and wait for help", "Climb out the window to the roof", "Try to break the door down and escape"]
    },
    "turn_2_C": {
        "story": "You disregard the alert and go to sleep, thinking it's all a dream. However, you wake up to the sound of zombies breaking into your house. You try to escape, but you are bitten and infected. You have failed to survive.",
        "choices": [],
        Status: "dead"
    } 
def story_stage(choices_made):
    """Tell Gemini where we are in the story so it doesn't rush the ending."""
    left = CHOICES_TO_SURVIVE - choices_made
    progress = choices_made / CHOICES_TO_SURVIVE
    if progress < 0.4:
        return f"STAGE: beginning. The goal is still far away. {left} choices left."
    if progress < 0.75:
        return f"STAGE: middle. Things get worse; reveal the twist soon. {left} choices left."
    return (f"STAGE: close to the goal, but NOT there yet. Add a final obstacle. "
            f"{left} choices left. Do NOT let the player escape yet.")


def make_choice(choice):
    """The player clicked a choice. OUR code decides what happens, not Gemini."""
    choices_made = len(st.session_state.picks) + 1
    damage = 0

    if choice.outcome == "death":
        result = "The player DIED because of this choice."
        ending = "dead"
    elif choice.outcome == "hurt":
        damage = random.randint(*DAMAGE[st.session_state.game["difficulty"]])
        new_health = max(0, st.session_state.health - damage)
        if new_health == 0:
            result = f"The player was hurt (-{damage} health) and DIED from their injuries."
            ending = "dead"
        else:
            result = f"The player got hurt and lost {damage} health (now {new_health})."
            ending = None
    else:
        result = "Nothing bad happened. The player is safe for now."
        ending = None

    # Last choice and still alive -> they made it!
    if ending is None and choices_made >= CHOICES_TO_SURVIVE:
        result += " This was the final choice: the player SURVIVED and escapes."
        ending = "survived"

    message = (
        f"I choose: {choice.text}\n"
        f"OUTCOME: {result}\n"
        f"This was choice {choices_made} of {CHOICES_TO_SURVIVE}."
    )
    if ending is None:
        message += "\n" + story_stage(choices_made) + " Give 3 new choices."

    if take_turn(message, ending):
        st.session_state.health = max(0, st.session_state.health - damage)
        st.session_state.picks.append({"text": choice.text, "outcome": choice.outcome, "damage": damage})


# =====================================================================
# 5. END-SCREEN POP-UPS (pictures are drawn with SVG code, no image files needed)
# =====================================================================

POPUP_STYLE = """
<style>
.end-art { text-align: center; animation: pop 0.6s ease-out; }
.end-art h1 { font-size: 2.6rem; margin: 0.3rem 0 0 0; letter-spacing: 2px; }
@keyframes pop { 0% { transform: scale(0.2); opacity: 0; }
                 70% { transform: scale(1.1); opacity: 1; }
                 100% { transform: scale(1); } }
.shake { animation: shake 0.4s ease-in-out 0.6s 3; transform-origin: center; }
@keyframes shake { 0%,100% { transform: rotate(0deg); }
                   25% { transform: rotate(-4deg); } 75% { transform: rotate(4deg); } }
.fly { animation: fly 2s ease-in-out infinite; }
@keyframes fly { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
</style>
"""

ZOMBIE_SVG = """
<div class="end-art">
<svg class="shake" width="240" height="240" viewBox="0 0 200 200">
  <rect x="80" y="150" width="40" height="40" fill="#5b4b3a"/>
  <rect x="35" y="140" width="130" height="30" rx="14" fill="#4a6b3a"/>
  <rect x="20" y="132" width="50" height="16" rx="8" fill="#7fae5a"/>
  <rect x="130" y="132" width="50" height="16" rx="8" fill="#7fae5a"/>
  <ellipse cx="100" cy="85" rx="58" ry="62" fill="#7fae5a"/>
  <path d="M60 50 Q75 30 95 42 Q110 25 130 45" stroke="#3d5a2a" stroke-width="6" fill="none"/>
  <circle cx="78" cy="78" r="16" fill="#f2f2d0"/>
  <circle cx="124" cy="82" r="11" fill="#f2f2d0"/>
  <circle cx="80" cy="80" r="6" fill="#b3001b"/>
  <circle cx="122" cy="84" r="4" fill="#b3001b"/>
  <path d="M68 112 Q100 128 132 110" stroke="#2b2b2b" stroke-width="5" fill="#2b2b2b"/>
  <rect x="84" y="112" width="8" height="9" fill="#f2f2d0"/>
  <rect x="104" y="114" width="8" height="8" fill="#f2f2d0"/>
  <path d="M135 60 l12 6 M138 70 l12 4" stroke="#3d5a2a" stroke-width="3"/>
</svg>
<h1 style="color:#d7263d;">YOU DIED</h1>
</div>
"""

RESCUE_SVG = """
<div class="end-art">
<svg width="260" height="200" viewBox="0 0 260 200">
  <circle cx="130" cy="170" r="70" fill="#ffb347"/>
  <rect x="0" y="160" width="260" height="40" fill="#2f4858"/>
  <rect x="30" y="120" width="30" height="40" fill="#33658a"/>
  <rect x="190" y="105" width="40" height="55" fill="#33658a"/>
  <g class="fly">
    <rect x="70" y="30" width="120" height="6" rx="3" fill="#333"/>
    <rect x="127" y="34" width="6" height="12" fill="#333"/>
    <ellipse cx="130" cy="62" rx="40" ry="20" fill="#4f772d"/>
    <ellipse cx="148" cy="58" rx="14" ry="10" fill="#a8dadc"/>
    <rect x="165" y="58" width="45" height="7" rx="3" fill="#4f772d"/>
    <rect x="205" y="48" width="6" height="20" rx="3" fill="#333"/>
    <rect x="105" y="80" width="50" height="4" rx="2" fill="#333"/>
  </g>
</svg>
<h1 style="color:#2a9d8f;">YOU SURVIVED!</h1>
</div>
"""


def restart_game():
    st.session_state.clear()
    st.rerun()


def show_end_clip(filename, fallback_drawing, title, color):
    """Show the clip from the assets folder, or the built-in drawing if it's missing."""
    path = ASSETS_FOLDER / filename
    if path.exists():
        if path.suffix.lower() in (".mp4", ".mov", ".webm"):
            st.video(str(path), autoplay=True, loop=True, muted=True)
        else:
            st.image(str(path))
        st.html(POPUP_STYLE + f'<div class="end-art"><h1 style="color:{color};">{title}</h1></div>')
    else:
        st.html(POPUP_STYLE + fallback_drawing)


@st.dialog("💀 Game over")
def death_popup():
    show_end_clip(LOSE_CLIP, ZOMBIE_SVG, "YOU DIED", "#d7263d")
    st.write(f"You lasted **{len(st.session_state.picks)}** of {CHOICES_TO_SURVIVE} choices.")
    if st.button("🔄 Play again", type="primary", use_container_width=True, key="popup_again_dead"):
        restart_game()


@st.dialog("🚁 Rescued!")
def win_popup():
    show_end_clip(WIN_CLIP, RESCUE_SVG, "YOU SURVIVED!", "#2a9d8f")
    st.write(f"You made all **{CHOICES_TO_SURVIVE}** choices and escaped the city.")
    if st.button("🔄 Play again", type="primary", use_container_width=True, key="popup_again_win"):
        restart_game()


# =====================================================================
# 6. THE WEB PAGE
# =====================================================================

st.set_page_config(page_title=GAME_TITLE, page_icon="🧟")
st.title(GAME_TITLE)

# Show any error from the last click (stays on screen after the page reloads)
if "error" in st.session_state:
    st.error(f"⚠️ Gemini had a problem: {st.session_state.pop('error')}")
if not get_api_key():
    st.warning("No API key found. Set GEMINI_API_KEY in the terminal before running the app.")

# First visit: set up empty game memory
if "history" not in st.session_state:
    st.session_state.history = []   # what we send to Gemini
    st.session_state.turns = []     # what we show on screen
    st.session_state.picks = []     # what the player picked + what happened
    st.session_state.health = 100
    st.session_state.ending = None  # None, "dead", or "survived"

# ----- START SCREEN -----
if not st.session_state.turns:
    st.write(
        "The outbreak started this morning. The city is falling. "
        f"Make **{CHOICES_TO_SURVIVE} choices** and stay alive to reach the rescue point. "
        "Every turn, one choice is safe, one will hurt you, and one will kill you. Choose carefully."
    )
    name = st.text_input("Your survivor's name", "Alex")
    if st.button("Start the game", type="primary"):
        st.session_state.game = pick_this_game()
        st.session_state.prompt = build_prompt(st.session_state.game)
        take_turn(f"Start the game. The player's name is {name}. Write the opening scene.")
        st.rerun()

# ----- GAME SCREEN -----
else:
    current = st.session_state.turns[-1]
    choices_made = len(st.session_state.picks)

    # Sidebar: health + progress + this game's settings
    with st.sidebar:
        st.header("Survivor status")
        st.progress(st.session_state.health / 100, text=f"❤️ Health: {st.session_state.health}")
        st.progress(choices_made / CHOICES_TO_SURVIVE,
                    text=f"🚁 Choices: {choices_made} / {CHOICES_TO_SURVIVE}")
        game = st.session_state.game
        st.caption(f"📍 {game['starting_location']}")
        st.caption(f"🕒 {game['time_of_day']} · {game['weather']}")
        st.caption(f"🧍 Companion: {game['companion']}")
        st.caption(f"⚔️ Difficulty: {game['difficulty']}")

    # Show the story so far
    for i, turn in enumerate(st.session_state.turns):
        st.write(turn.story)
        if i < len(st.session_state.picks):
            pick = st.session_state.picks[i]
            if pick["outcome"] == "hurt":
                st.warning(f"➡️ {pick['text']}  🩸 -{pick['damage']} health")
            elif pick["outcome"] == "death":
                st.error(f"➡️ {pick['text']}  💀")
            else:
                st.info(f"➡️ {pick['text']}  😮‍💨 close call, no damage")

    # Game over: show the pop-up once, then only a Play again button
    if st.session_state.ending:
        if not st.session_state.get("popup_shown"):
            st.session_state.popup_shown = True
            if st.session_state.ending == "dead":
                death_popup()
            else:
                st.balloons()
                win_popup()
        # The clip also stays on the page after the pop-up is closed
        if st.session_state.ending == "dead":
            show_end_clip(LOSE_CLIP, ZOMBIE_SVG, "YOU DIED", "#d7263d")
            st.error(f"💀 You lasted {choices_made} choices.")
        else:
            show_end_clip(WIN_CLIP, RESCUE_SVG, "YOU SURVIVED!", "#2a9d8f")
            st.success("🚁 You made it to the rescue point.")
        if st.button("🔄 Play again", type="primary"):
            restart_game()

    # Still alive: show the 3 choices
    else:
        st.subheader("What do you do?")
        for i, choice in enumerate(current.choices):
            if st.button(choice.text, use_container_width=True, key=f"choice_{choices_made}_{i}"):
                make_choice(choice)
                st.rerun()