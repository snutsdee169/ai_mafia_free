import time
import random
import re
from termcolor import colored
from dataclasses import dataclass, field
from typing import List, Optional, Union
from google import genai 

# --- CONFIGURATION ---
CONFIG = {
    # Replace with your actual key
    "api_key": "AIzaSyBoHKLKTLVrixkoQmZ8sp4wSluvZ6RjGkc", 
    
    # Game Settings
    "discussion_rounds_per_day": 2,
    "mafia_discussion_rounds_per_night": 2,
    "max_retries": float("inf"),
    
    # Model Assignments 
    "models": {
        "mafia": "gemma-3-27b-it",
        "doctor": "gemma-3-27b-it",
        "detective": "gemma-3-27b-it",
        "town": "gemma-3-27b-it"
    },

    "rules": """
    You are playing a game of Mafia. 
    - The Mafia wins if their number equals the number of Innocents.
    - The Innocents (Town, Doctor, Detective) win if all Mafia are eliminated.
    - Day Phase: Players discuss and vote to eliminate one suspect.
    - Night Phase: 
        - Mafia kills one person.
        - Doctor saves one person (can save self, cannot save same person twice in a row).
        - Detective investigates one person to learn their role.
    - If you're asked for analysis but it's only day/night 1, simply say that nothing's happened so you haven't got anything to say.
    - For smooth automation of the game, do not appempt to send a private note to the moderator during discussion time.
    """
}

ROLE_COLORS = {
    "Mafia": "red",
    "Doctor": "blue",
    "Detective": "magenta", # Termcolor uses magenta for purple
    "Town": "white"
}

# --- CLASSES ---

@dataclass
class Player:
    name: str
    role: str
    model_id: str
    is_alive: bool = True
    is_protected: bool = False
    last_protected_target: str = None 
    private_memory: List[str] = field(default_factory=list)

    @property
    def colored_name(self) -> str:
        """Returns the player's name colored based on their role."""
        color = ROLE_COLORS.get(self.role, "white")
        return colored(self.name, color)

    def __str__(self):
        return f"{self.name} ({self.role})"


class LLMInterface:
    """Handles communication with the Google GenAI API."""
    def __init__(self, api_key):
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"❌ Error initializing GenAI Client: {e}")
            self.client = None

    def generate(self, player: Player, context: str, instruction: str, distinct_action: bool = False, all_players: List[Player] = None) -> str:
        if not self.client:
            return "Error: Client not initialized."
        
        # --- FIX 1: Clean list of alive players ---
        if all_players:
            others_list = [
                p.name for p in all_players 
                if p.is_alive and p.name != player.name
            ]
            others_str = ", ".join(others_list)
        else:
            others_str = "Unknown"

        # --- FIX 2: Clean list of Mafia teammates ---
        mafia_info = ""
        if player.role == "Mafia" and all_players:
            teammates = [
                p.name for p in all_players 
                if p.role == "Mafia" and p.name != player.name
            ]
            mafia_info = f"Your fellow Mafia members are: {', '.join(teammates)}"

# --- IN THE LLMInterface.generate METHOD ---

# 1. Define Role-Specific Strategic Advice
        role_strategy = ""
        if player.role == "Mafia":
            role_strategy = (
                "- Be deceptive. Blend in with the Town by acting concerned about the deaths.\n"
                "- Coordinate subtly. If your teammate is under fire, decide whether to defend them or 'bus' them (vote them out) to look innocent.\n"
                "- Avoid repeating the exact phrasing of your fellow Mafia members."
            )
        elif player.role == "Detective":
            role_strategy = (
                "- You have the most power. If you find a Mafia member, you must convince the town to vote them out.\n"
                "- You can choose to 'claim' your role if necessary, but remember the Mafia will target you at night if you do.\n"
                "- Use your investigation results as 'strong hunches' or 'certainty' to lead the Town."
            )
        elif player.role == "Doctor":
            role_strategy = (
                "- Stay alive. You are the only thing keeping the power roles (Detective) safe.\n"
                "- Pay attention to who is leading the discussion; they are likely the Detective or a target for the Mafia."
            )
        else: # Town
            role_strategy = (
                "- Look for contradictions and patterns. Who voted for the person who turned out to be an Innocent?\n"
                "- If someone is making a very specific accusation, consider if they might be the Detective before calling them 'suspicious'.\n"
                "- Do not be passive. The Mafia wins if you don't find them."
            )

        # 2. Construct the New System Prompt
        system_prompt = (
            f"### ROLE IDENTITY\n"
            f"You are {player.name}, and your secret role is {player.role.upper()}.\n"
            f"Your winning condition: {'Eliminate all Innocents' if player.role == 'Mafia' else 'Eliminate all Mafia members'}.\n\n"
            
            f"### STRATEGIC MANDATE\n"
            f"{role_strategy}\n"
            f"- **Survival Instinct:** CRITICAL: You are {player.name}. You are currently alive. Do not, under any circumstances, vote to eliminate yourself. If you think you should vote for {player.name}, you are confused -- you are {player.name}!\n"
            f"- **Linguistic Diversity:** Do NOT parrot or mimic the phrases used by other players. If others are saying 'this is concerning,' use different language like 'I'm looking at the facts' or 'Something doesn't add up about X'.\n"
            f"- **Critical Thinking:** Don't just vote because someone is 'quiet' or 'loud.' Look at their voting history.\n\n"
            
            f"### CURRENT GAME STATE\n"
            f"- {mafia_info}\n"
            f"- Players currently alive: {others_str}\n"
            f"- Rules: {CONFIG['rules']}\n"
            f"- History of events:\n{context}\n"
            f"{self._get_private_context(player)}"
        )

        if distinct_action:
            system_prompt += "\n🔴 IMPORTANT: Your response must be EXACTLY just the name of the target player. Do not write sentences."

        full_prompt = f"{system_prompt}\n\nINSTRUCTION: {instruction}"

        while True:
            try:
                response = self.client.models.generate_content(
                    model=player.model_id,
                    contents=full_prompt
                )
                return response.text.strip() if response.text else ""
                
            except Exception as e:
                time.sleep(5)

    def _get_private_context(self, player: Player) -> str:
        if player.private_memory:
            memory_log = "\n".join(player.private_memory)
            return f"\n🧠 YOUR INTERNAL MONOLOGUE & NOTES:\n{memory_log}\n"
        return ""


class GameEngine:
    def __init__(self):
        self.llm = LLMInterface(CONFIG["api_key"])
        self.players = self._setup_players()
        self.shared_history = ["--- GAME START ---"]
        self.day_count = 0
        self.is_game_over = False

    def _setup_players(self) -> List[Player]:
        return [
            Player("Madison", "Mafia", CONFIG["models"]["mafia"]),
            Player("Avery", "Mafia", CONFIG["models"]["mafia"]),
            Player("Anna", "Mafia", CONFIG["models"]["mafia"]),
            Player("Maddie", "Doctor", CONFIG["models"]["doctor"]),
            Player("Valentina", "Detective", CONFIG["models"]["detective"]),
            Player("Amber", "Town", CONFIG["models"]["town"]),
            Player("Elena", "Town", CONFIG["models"]["town"]),
            Player("Ariana", "Town", CONFIG["models"]["town"]),
            Player("Sava", "Town", CONFIG["models"]["town"]),
            Player("Summer", "Town", CONFIG["models"]["town"]),
            Player("Natalie", "Town", CONFIG["models"]["town"]),
        ]

    def _strip_ansi(self, text: str) -> str:
        """Removes ANSI escape codes from a string for clean history logging."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def log(self, message: str, to_console: bool = True, to_shared_history: bool = True):
        """Adds to internal history (clean) and prints to CLI (colored)."""
        if to_shared_history:
            self.shared_history.append(self._strip_ansi(message))
        if to_console:
            print(f"{message}")

    def get_player_by_name(self, name: str) -> Optional[Player]:
        name = name.lower().strip()
        name = name.replace('.', '').replace('!', '')
        for p in self.players:
            if p.name.lower() in name:
                return p
        return None

    def _get_valid_action_response(self, actor: Player, prompt: str, candidates: List[str]) -> Union[Player, str, None]:
        candidate_str = ", ".join(candidates)
        base_prompt = f"{prompt} Valid targets are: [{candidate_str}]."
        current_prompt = base_prompt
        
        while True:
            response = self.llm.generate(actor, "\n".join(self.shared_history), current_prompt, distinct_action=True, all_players=self.players)
            if "Skip" in candidates and response.lower() == "skip":
                return "Skip"
            target = self.get_player_by_name(response)
            
            if target and target.name in candidates:
                return target
            
            print(f"      [!] Retrying {actor.colored_name} due to invalid output: '{response}'")
            current_prompt = (
                f"{base_prompt}\n\n"
                f"SYSTEM ERROR: You replied '{response}', which is NOT a valid target. "
                f"Please output ONLY a name from this exact list: [{candidate_str}]."
            )

    def _get_inner_thoughts(self, actor: Player, prompt: str):
        response = self.llm.generate(
            actor, 
            "\n".join(self.shared_history), 
            f"{prompt} Keep it brief (1 sentence). NONE OF THE PLAYERS will see your response to this message, so answer in accordance to your true intentions and your role.",
            all_players=self.players
        )
        actor.private_memory.append(f"Thought (Day {self.day_count}): {response}")
        self.log(f"  💭 [{actor.colored_name} thinking]: {response}", to_console=True, to_shared_history=False)

    # --- NIGHT PHASE HELPERS ---

    def _night_mafia_phase(self, alive_players, alive_names) -> Optional[Player]:
        alive_mafia = [p for p in alive_players if p.role == "Mafia"]
        if not alive_mafia:
            return None

        if len(alive_mafia) > 1:
            conspirators = ', '.join(p.colored_name for p in alive_mafia)
            self.log(f"\n  🌑 The Mafia ({conspirators}) are conspiring...", to_shared_history=False)
            mafia_chat_history = []
            
            for round_num in range(CONFIG["mafia_discussion_rounds_per_night"]):
                for m_player in alive_mafia:
                    chat_context = "\n".join(mafia_chat_history[-4:]) 
                    statement = self.llm.generate(
                        m_player,
                        f"Night {self.day_count} Mafia Chat:\n{chat_context}",
                        "Discuss with your fellow Mafia members who to kill tonight. Be strategic. Keep it brief (1 sentence).",
                        all_players=self.players
                    )
                    
                    msg = f"Night {self.day_count} - {m_player.name} (Mafia Chat): {statement}"
                    mafia_chat_history.append(msg)
                    self.log(f"    💬 [{m_player.colored_name}]: {statement}", to_shared_history=False)
                    
                    for obs in alive_mafia:
                        obs.private_memory.append(msg)

            mafia_killer = random.choice(alive_mafia)
            self.log(f"  🔪 {mafia_killer.colored_name} steps forward to perform the hit.", to_shared_history=False)
        else:
            mafia_killer = alive_mafia[0]
            self.log(f"\n  🌑 {mafia_killer.colored_name} is choosing a target...", to_shared_history=False)
            self._get_inner_thoughts(mafia_killer, "Who is the biggest threat to you right now?")

        target_kill = self._get_valid_action_response(
            mafia_killer, 
            "Who do you want to KILL tonight?", 
            [n for n in alive_names if n != mafia_killer.name]
        )
        if target_kill:
            self.log(f"  🎯 The Mafia has targeted {target_kill.colored_name}.", to_shared_history=False)
            
        return target_kill

    def _night_doctor_phase(self, alive_players, alive_names):
        doctor = next((p for p in alive_players if p.role == "Doctor"), None)
        if not doctor:
            return

        self.log(f"\n  ⚕️  {doctor.colored_name} is choosing a patient...", to_shared_history=False)
        valid_saves = [n for n in alive_names if n != doctor.last_protected_target]
        
        self._get_inner_thoughts(doctor, "Who will you save tonight and why?")
        target_save = self._get_valid_action_response(
            doctor, 
            "Who do you want to SAVE tonight?", 
            valid_saves
        )
        
        if target_save and isinstance(target_save, Player):
            target_save.is_protected = True
            self.log(f"  🛡️  The Doctor is protecting {target_save.colored_name}.", to_shared_history=False)
            doctor.last_protected_target = target_save.name

    def _night_detective_phase(self, alive_players, alive_names):
        detective = next((p for p in alive_players if p.role == "Detective"), None)
        if not detective:
            return

        self.log(f"\n  🔎 {detective.colored_name} is investigating a suspect...", to_shared_history=False)
        self._get_inner_thoughts(detective, "Who will you investigate tonight and why?")
        
        target_investigate = self._get_valid_action_response(
            detective, 
            "Who do you want to INVESTIGATE?", 
            [n for n in alive_names if n != detective.name]
        )
        
        if target_investigate and isinstance(target_investigate, Player):
            result = "Mafia" if target_investigate.role == "Mafia" else "Innocent"
            directive = f"Do everything you can to eliminate {target_investigate.name}" if target_investigate.role == 'Mafia' else "Remember that this person is innocent."
            note = f"Night {self.day_count}: Investigated {target_investigate.name}. Result: {result}. {directive}"
            detective.private_memory.append(note)
            self.log(f"  🔍 (Detective discovered that {target_investigate.colored_name} is {result})", to_shared_history=False)

    # --- PHASES ---

    def run_night_phase(self):
        self.day_count += 1
        self.log(f"\n{'='*40}\n🌙  NIGHT {self.day_count}\n{'='*40}")

        for p in self.players:
            p.is_protected = False

        alive_players = [p for p in self.players if p.is_alive]
        alive_names = [p.name for p in alive_players]

        target_kill = self._night_mafia_phase(alive_players, alive_names)
        self._night_doctor_phase(alive_players, alive_names)
        self._night_detective_phase(alive_players, alive_names)

        self.log(f"\n{'-'*40}\n🌅  MORNING REPORT\n{'-'*40}")
        if target_kill and isinstance(target_kill, Player):
            if target_kill.is_protected:
                self.log(f"  🩸 Mafia attacked {target_kill.colored_name}, but they were saved by the Doctor!")
            else:
                target_kill.is_alive = False
                self.log(f"  🪦 {target_kill.colored_name} was found dead this morning.")
        else:
            self.log("  🕊️ The night was quiet. No one died.")

    def run_day_phase(self):
        self.log(f"\n{'='*40}\n☀️  DAY {self.day_count}\n{'='*40}")
        
        if self.check_win_condition(): return

        alive = [p for p in self.players if p.is_alive]
        
        # 1. Discussion Rounds
        for round_num in range(CONFIG["discussion_rounds_per_day"]):
            self.log(f"\n  {'-'*10} Discussion Round {round_num + 1} {'-'*10}\n")
            for player in alive:
                statement = self.llm.generate(
                    player, 
                    "\n".join(self.shared_history), 
                    "Discuss the recent events and who you suspect. Keep it under 2 sentences. YOUR RESPONSE HERE WILL NOT BE A PRIVATE MESSAGE TO OTHER MAFIA. EVERY PLAYER WILL SEE YOUR RESPONSE TO THIS ONE. DO NOT ADD ANY SELF-INCRIMINATING COMMENTS OR THOUGHTS TO YOUR RESPONSE. Detective, reference your notes when deciding who to discuss/accuse for.",
                    all_players=self.players
                )
                self.log(f"  🗣️ [{player.colored_name}]: {statement}")

        # 2. Voting
        self.log(f"\n  {'-'*10} VOTING PHASE {'-'*10}\n")
        votes = {}
        
        for player in alive:
            self._get_inner_thoughts(player, "Who do you want to vote for and why?")
            
            vote_target = self._get_valid_action_response(
                player,
                "Remember that your votes can be seen publicly. Detective, reference your notes when deciding who to vote for. Who do you vote to eliminate? Output player name or 'Skip'.",
                [p.name for p in alive] + ['Skip']
            )
            
            if vote_target == "Skip":
                votes["Skip"] = votes.get("Skip", 0) + 1
                self.log(f"  🤚 {player.colored_name} voted to Skip.")
            elif vote_target:
                votes[vote_target.name] = votes.get(vote_target.name, 0) + 1
                self.log(f"  ⚖️ {player.colored_name} voted for {vote_target.colored_name}.")
            else:
                self.log(f"  🤚 {player.colored_name} abstained.")

        # 3. Resolve Vote
        self.log(f"\n  {'-'*10} VOTE RESOLUTION {'-'*10}\n")
        if votes:
            max_votes = max(votes.values())
            candidates = [name for name, count in votes.items() if count == max_votes]
            
            if len(candidates) == 1:
                victim_name = candidates[0]
                if victim_name == "Skip":
                    self.log("  🏳️ The town voted to Skip. No one was executed.")
                else:
                    victim = self.get_player_by_name(victim_name)
                    if victim:
                        victim.is_alive = False
                        self.log(f"  🪓 The town has decided. {victim.colored_name} is executed.\n  🎭 Their role was: {victim.role}")
            else:
                self.log("  ⚖️ The vote was tied. No one was executed.")
        else:
            self.log("  ⚖️ No votes were cast.")

    def check_win_condition(self) -> bool:
        mafia = [p for p in self.players if p.role == "Mafia" and p.is_alive]
        innocent = [p for p in self.players if p.role != "Mafia" and p.is_alive]

        if not mafia:
            self.log(f"\n{'='*40}\n🏆 GAME OVER: The Innocents have won!\n{'='*40}")
            self.is_game_over = True
            input(">")
            return True
        
        if len(mafia) >= len(innocent):
            self.log(f"\n{'='*40}\n🏆 GAME OVER: The Mafia has taken over the town!\n{'='*40}")
            self.is_game_over = True
            input(">")
            return True
        
        return False

    def start(self):
        print("\nStarting Mafia Simulation...\n")
        while not self.is_game_over:
            self.run_night_phase()
            if self.check_win_condition(): break
            self.run_day_phase()
            if self.check_win_condition(): break

# --- EXECUTION ---
if __name__ == "__main__":
    if CONFIG["api_key"] == "YOUR_API_KEY_HERE":
        print("❌ Please update the CONFIG dictionary with your API Key.")
    else:
        game = GameEngine()
        game.start()
