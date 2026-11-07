"""
🧟 Zombie Survival — a choose-your-own-adventure game powered by Gemini
Run with:  streamlit run app.py
"""

import os
import random
import time
from typing import Literal

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel


# =====================================================================
# 1. SETTINGS  — the easiest things to change are all up here
# =====================================================================

GAME_TITLE = "🧟 Zombie Survival"
# Models to try, in order. The "lite" models are the fastest, so they go first.
# If one is busy (503), too slow, or missing (404), the game tries the next one.
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]
SECONDS_PER_TRY = 20   # give up on a model after this many seconds and try the next
CHOICES_TO_SURVIVE = 5         # make this many choices without dying = you survive


# ---------------------------------------------------------------------
# YOUR STORY SETTINGS  —  change these to control the story.
# Single values ("text") are used every game.
# Lists ([...]) -> the game randomly picks ONE item each new game,
# so every playthrough is different. Add as many options as you want!
# ---------------------------------------------------------------------

STORY_SETTINGS = {
    # --- the same every game ---
    "city": "Buffalo, New York",
    "goal": "reach the military rescue helicopter waiting on the roof of a hospital downtown",
    "zombie_type": "slow, but they travel in huge hordes and are attracted to noise",
    "tone": "tense and scary, with a little dark humor",
    "difficulty": "medium",   # pick one: "easy", "medium", "hard"

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

# What each difficulty means (you can edit these too)
DIFFICULTY_RULES = {
    "easy":   "Bad choices lower health by 5 to 20. Instant death is very rare.",
    "medium": "Bad choices lower health by 10 to 40. A truly reckless choice can kill instantly.",
    "hard":   "Bad choices lower health by 20 to 50. Deadly choices are common, so the player must think carefully.",
}


def pick_this_game():
    """Lock in the settings for ONE game. Lists get a random pick."""
    game = {}
    for key, value in STORY_SETTINGS.items():
        if isinstance(value, list):
            game[key] = random.choice(value)
        else:
            game[key] = value
    return game


def build_prompt(game):
    """Turn this game's settings into the rules Gemini follows."""
    return f"""
You are the narrator of a zombie survival choose-your-own-adventure game.
Invent a fresh, original story every time. Do not reuse plots.

STORY SETTINGS FOR THIS GAME (follow these exactly):
- City: {game["city"]}
- The player starts in: {game["starting_location"]}
- Time of day: {game["time_of_day"]}
- Weather: {game["weather"]}
- Companion: {game["companion"]}
- Zombies: {game["zombie_type"]}
- Goal: {game["goal"]}
- Tone: {game["tone"]}
- SECRET TWIST (never tell the player directly; hint at it, then reveal it
  somewhere in the second half of the story): {game["secret_twist"]}

HOW TO WRITE EACH TURN:
- Write in second person ("You hear something scratching at the door...").
- Write 1 short paragraph (3-5 sentences). No graphic gore.
- Then give exactly 3 choices. Each choice is one short sentence.
- Every turn, at least one choice should be dangerous, and the
  choices should lead to really different outcomes.
- Remember everything that happened earlier and stay consistent.

HEALTH:
- The player starts at 100 health (max 100).
- {DIFFICULTY_RULES[game["difficulty"]]}
- Smart choices can keep health the same or raise it a little.

ENDINGS (there are only two):
1. "dead"     -> health reaches 0 or the player makes a deadly choice.
                 Write a dramatic death scene and give NO choices.
2. "survived" -> ONLY allowed when you are told it is the FINAL TURN.
                 Write an exciting escape ending and give NO choices.
- Otherwise the status is "alive" and you give 3 choices.
- The game lasts {CHOICES_TO_SURVIVE} choices. Slowly move the player toward
  the goal so the ending makes sense when they get there.
"""


# The 3 possible statuses in the game
Status = Literal["alive", "dead", "survived"]


# =====================================================================
# 2. THE SHAPE OF GEMINI'S ANSWER (structured output)
#    Gemini ALWAYS replies with JSON that matches this.
# =====================================================================

class Turn(BaseModel):
    story: str
    choices: list[str]
    health: int
    status: Status


# =====================================================================
# 3. TALKING TO GEMINI
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
        temperature=1.0,  # higher = more creative / random stories
    )
    if low_thinking:
        config.thinking_config = types.ThinkingConfig(thinking_level="LOW")  # faster answers

    response = client.models.generate_content(model=model, contents=history, config=config)

    # Normally Gemini gives us a ready-made Turn in response.parsed.
    # If not, try reading the raw JSON text ourselves.
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


def take_turn(player_message):
    """Player does something -> Gemini answers -> save everything."""
    add_message("user", player_message)
    try:
        with st.spinner("Something is moving in the dark... "):
            turn = ask_gemini(st.session_state.history)
    except Exception as error:
        st.session_state.history.pop()  # undo so the player can try again
        # Save the error so it still shows after the page reloads
        st.session_state.error = str(error)
        return False

    # Safety rules in our own code, in case Gemini forgets:
    if turn.health <= 0:
        turn.status = "dead"
        turn.health = 0
    if turn.status != "alive":
        turn.choices = []

    add_message("model", turn.model_dump_json())
    st.session_state.turns.append(turn)
    return True


def make_choice(choice):
    """Called when the player clicks a choice (or types their own)."""
    choices_made = len(st.session_state.picks) + 1

    if choices_made >= CHOICES_TO_SURVIVE:
        message = (
            f"I choose: {choice}. "
            f"This is choice {choices_made} of {CHOICES_TO_SURVIVE}. FINAL TURN: "
            f"if this choice is deadly, the status is 'dead'. Otherwise the player "
            f"reaches the rescue point and the status is 'survived'."
        )
    else:
        message = (
            f"I choose: {choice}. "
            f"This is choice {choices_made} of {CHOICES_TO_SURVIVE}. "
            f"Do NOT end with 'survived' yet."
        )

    if take_turn(message):
        st.session_state.picks.append(choice)


# =====================================================================
# 4. THE WEB PAGE
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
    st.session_state.picks = []     # the choices the player clicked

# ----- START SCREEN -----
if not st.session_state.turns:
    st.write(
        "The outbreak started this morning. The city is falling. "
        f"Make **{CHOICES_TO_SURVIVE} choices** and stay alive to reach the rescue point."
    )
    name = st.text_input("Your survivor's name", "Alex")
    if st.button("Start the game", type="primary"):
        st.session_state.game = pick_this_game()
        st.session_state.prompt = build_prompt(st.session_state.game)
        take_turn(
            f"Start the game. The player's name is {name}. "
            f"Health is 100. Status is 'alive'. This is the opening scene."
        )
        st.rerun()

# ----- GAME SCREEN -----
else:
    current = st.session_state.turns[-1]
    choices_made = len(st.session_state.picks)

    # Sidebar: health + progress
    with st.sidebar:
        st.header("Survivor status")
        st.progress(max(0, min(current.health, 100)) / 100, text=f"❤️ Health: {current.health}")
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
            st.info(f"➡️ You chose: {st.session_state.picks[i]}")

    # Ending screens
    if current.status == "dead":
        st.error(f"💀 YOU DIED after {choices_made} choices.")
    elif current.status == "survived":
        st.success("🚁 YOU SURVIVED! You made it to the rescue point.")
        st.balloons()

    # Still alive: show choices
    else:
        st.subheader("What do you do?")
        for choice in current.choices:
            if st.button(choice, use_container_width=True):
                make_choice(choice)
                st.rerun()

        custom = st.text_input("...or type your own action")
        if st.button("Do it") and custom:
            make_choice(custom)
            st.rerun()

    if st.button("🔄 Play again"):
        st.session_state.clear()
        st.rerun()
