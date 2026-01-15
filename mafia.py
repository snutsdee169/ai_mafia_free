# Created by deesnuts69

import os
import time
import random  # Added for random selection of mafia killer
from dataclasses import dataclass, field
from typing import List, Optional, Union
from google import genai 

# --- CONFIGURATION ---
CONFIG = {
    # Replace with your actual key
    "api_key": "ADD API KEY HERE", 
    
    # Game Settings
    "discussion_rounds_per_day": 2,
    "mafia_discussion_rounds_per_night": 1, # New setting
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
    - If you're asked for analuysis but it's only day/night 1, simply say that nothing's happened so you haven't got anything to say.
    - For smooth automation of the game, do not appempt to send a private note to the moderator during discussion time.
    """
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
            # Get names of other mafia members (excluding self)
            teammates = [
                p.name for p in all_players 
                if p.role == "Mafia" and p.name != player.name
            ]
            mafia_info = f"Your fellow Mafia members are: {', '.join(teammates)}"

        # 2. Construct the System Prompt
        system_prompt = (
            f"You are {player.name}, assigned the role of {player.role}.\n"
            f"{mafia_info}\n"  # Insert the clean string here
            f"The current list of other players who are alive: {others_str}\n" 
            f"Game Rules: {CONFIG['rules']}\n"
            f"Current Context:\n{context}\n"
            f"{self._get_private_context(player)}\n"
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
                # On success, return the value (this breaks the loop)
                return response.text.strip() if response.text else ""
                
            except Exception as e:
                time.sleep(5)
                # The loop will now restart automatically

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
            Player("Brooke", "Mafia", CONFIG["models"]["mafia"]),
            Player("Maddie", "Doctor", CONFIG["models"]["doctor"]),
            Player("Valentina", "Detective", CONFIG["models"]["detective"]),
            Player("Amber", "Town", CONFIG["models"]["town"]),
            Player("Elena", "Town", CONFIG["models"]["town"]),
            Player("Ariana", "Town", CONFIG["models"]["town"]),
            Player("Tate", "Town", CONFIG["models"]["town"]),
            Player("Summer", "Town", CONFIG["models"]["town"]),
            Player("Natalie", "Town", CONFIG["models"]["town"]),
        ]

    def log(self, message: str, to_console: bool = True, to_shared_history: bool = True):
        """Adds to internal history and prints to CLI."""
        if to_shared_history:
            self.shared_history.append(message)
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
        """Loops until LLM returns a valid player name or retries run out."""
        candidate_str = ", ".join(candidates)
        base_prompt = f"{prompt} Valid targets are: [{candidate_str}]."
        current_prompt = base_prompt
        
        while True: # retry counter could go here
            response = self.llm.generate(actor, "\n".join(self.shared_history), current_prompt, distinct_action=True, all_players=self.players)
            if "Skip" in candidates and response.lower() == "skip":
                return "Skip"
            target = self.get_player_by_name(response)
            
            if target and target.name in candidates:
                return target
            
            print(f"   (Retrying {actor.name} due to invalid output: '{response}')")
            current_prompt = (
                f"{base_prompt}\n\n"
                f"SYSTEM ERROR: You replied '{response}', which is NOT a valid target. "
                f"Please output ONLY a name from this exact list: [{candidate_str}]."
            )
        
        print(f"   (Failed to get valid input from {actor.name}. Skipping action.)")
        return None

    def _get_inner_thoughts(self, actor: Player, prompt: str):
        response = self.llm.generate(
            actor, 
            "\n".join(self.shared_history), 
            f"{prompt} Keep it brief (1 sentence). NONE OF THE PLAYERS will see your response to this message, so answer in accordance to your true intentions and your role.",
            all_players=self.players
        )
        actor.private_memory.append(f"Thought (Day {self.day_count}): {response}")
        self.log(f"({actor.name}'s inner thought): {response}", to_console=True, to_shared_history=False)

    # --- PHASES ---

    def run_night_phase(self):
        #time.sleep(60) # Prevent running out of prompts per minute
        self.day_count += 1
        header = f"\n=== NIGHT {self.day_count} ===\n"
        self.log(header)

        # 1. Reset protection
        for p in self.players:
            p.is_protected = False

        alive_players = [p for p in self.players if p.is_alive]
        alive_names = [p.name for p in alive_players]

        # 2. Mafia Action
        alive_mafia = [p for p in alive_players if p.role == "Mafia"]
        target_kill = None
        
        if alive_mafia:
            mafia_killer = None
            
            # --- MULTI-MAFIA CHAT LOGIC ---
            if len(alive_mafia) > 1:
                print(f"🌑 The Mafia ({', '.join(p.name for p in alive_mafia)}) are conspiring...")
                mafia_chat_history = []
                
                for round_num in range(CONFIG["mafia_discussion_rounds_per_night"]):
                    for m_player in alive_mafia:
                        # Construct context for the chat (including what peers just said)
                        chat_context = "\n".join(mafia_chat_history[-4:]) # Keep recent context
                        
                        statement = self.llm.generate(
                            m_player,
                            f"Night {self.day_count} Mafia Chat:\n{chat_context}",
                            "Discuss with your fellow Mafia members who to kill tonight. Be strategic. Keep it brief (1 sentence).",
                            all_players=self.players
                        )
                        
                        # Format message
                        msg = f"Night {self.day_count} - {m_player.name} (Mafia Chat): {statement}"
                        mafia_chat_history.append(msg)
                        print(f"{m_player.name}: {statement}")
                        
                        # Add to EVERY mafia member's private memory so they remember the plan
                        for obs in alive_mafia:
                            obs.private_memory.append(msg)

                # Select a random mafia member to execute the kill
                mafia_killer = random.choice(alive_mafia)
                print(f"🌑 {mafia_killer.name} steps forward to perform the hit.")

            # --- SINGLE MAFIA LOGIC ---
            else:
                mafia_killer = alive_mafia[0]
                print(f"🌑 {mafia_killer.name} is choosing a target...")
                # Only use inner thoughts if alone (if group, they already discussed)
                self._get_inner_thoughts(mafia_killer, "Who is the biggest threat to you right now?")

            # Perform the Kill Action (by the selected killer)
            target_kill = self._get_valid_action_response(
                mafia_killer, 
                "Who do you want to KILL tonight?", 
                [n for n in alive_names if n != mafia_killer.name] # Can't kill self
            )
            self.log(f"🌑 The mafias will kill {target_kill.name}\n")

        # 3. Doctor Action
        doctor = next((p for p in alive_players if p.role == "Doctor"), None)
        target_save = None
        if doctor:
            print("⚕️ Doctor is choosing a patient...")
            valid_saves = [n for n in alive_names if n != doctor.last_protected_target]
            
            self._get_inner_thoughts(doctor, "Who will you save tonight and why?")
            target_save = self._get_valid_action_response(
                doctor, 
                "Who do you want to SAVE tonight?", 
                valid_saves
            )
            if target_save and isinstance(target_save, Player):
                target_save.is_protected = True
                self.log(f"⚕️ The doctor visited {target_save.name}\n")
                doctor.last_protected_target = target_save.name

        # 4. Detective Action
        detective = next((p for p in alive_players if p.role == "Detective"), None)
        if detective:
            print("🔎 Detective is investigating...")
            self._get_inner_thoughts(detective, "Who will you investigate tonight and why?")
            
            target_investigate = self._get_valid_action_response(
                detective, 
                "Who do you want to INVESTIGATE?", 
                [n for n in alive_names if n != detective.name]
            )
            if target_investigate and isinstance(target_investigate, Player):
                result = "Mafia" if target_investigate.role == "Mafia" else "Innocent"
                note = f"Night {self.day_count}: Investigated {target_investigate.name}. Result: {result}. {f"Do everything you can to eliminate {target_investigate.name}" if target_investigate.role == 'Mafia' else f"Remember that this person is innocent."}"
                detective.private_memory.append(note)
                print(f"🔎 (Detective found that {target_investigate.name} is {result})")

        # 5. Resolve Night
        self.log("\n--- MORNING REPORT ---")
        if target_kill and isinstance(target_kill, Player):
            if target_kill.is_protected:
                self.log(f"🌑 Mafia attacked {target_kill.name}, but they were saved by the Doctor!")
            else:
                target_kill.is_alive = False
                self.log(f"🪦 {target_kill.name} was found dead this morning.")
        else:
            self.log("The night was quiet.")

    def run_day_phase(self):
        #time.sleep(60) # Prevent running out of prompts per minute
        header = f"\n=== DAY {self.day_count} ===\n"
        self.log(header)
        
        if self.check_win_condition(): return

        alive = [p for p in self.players if p.is_alive]
        
        # 1. Discussion Rounds
        for round_num in range(CONFIG["discussion_rounds_per_day"]):
            self.log(f"\n--- Discussion Round {round_num + 1} ---")
            for player in alive:
                if not player.is_alive: continue 
                
                print(f"💬 {player.name} is thinking...")
                statement = self.llm.generate(
                    player, 
                    "\n".join(self.shared_history), 
                    "Discuss the recent events and who you suspect. Keep it under 2 sentences. YOUR RESPONSE HERE WILL NOT BE A PRIVATE MESSAGE TO OTHER MAFIA. EVERY PLAYER WILL SEE YOUR RESPONSE TO THIS ONE. DO NOT ADD ANY SELF-INCRIMINATING COMMENTS OR THOUGHTS TO YOUR RESPONSE. Detective, reference your notes when deciding who to discuss/accuse for.",
                    all_players=self.players
                )
                self.log(f"[{player.name}]: {statement}")
                print('\n')
            #time.sleep(60) # Prevent running out of prompts per minute
        # 2. Voting
        self.log("\n--- VOTING TIME ---")
        votes = {}
        
        for player in alive:
            print(f"🗳️ {player.name} is voting...")
            # Think first
            self._get_inner_thoughts(player, "Who do you want to vote for and why?")
            
            vote_target = self._get_valid_action_response(
                player,
                "Remember that your votes can be seen publicly. Detective, reference your notes when deciding who to vote for. Who do you vote to eliminate? Output player name or 'Skip'.",
                [p.name for p in alive] + ['Skip']
            )
            
            if vote_target == "Skip":
                votes["Skip"] = votes.get("Skip", 0) + 1
                self.log(f"{player.name} voted to Skip.")
            elif vote_target:
                votes[vote_target.name] = votes.get(vote_target.name, 0) + 1
                self.log(f"{player.name} voted for {vote_target.name}")
            else:
                self.log(f"{player.name} abstained.")
            #time.sleep(60)

        # 3. Resolve Vote
        if votes:
            max_votes = max(votes.values())
            candidates = [name for name, count in votes.items() if count == max_votes]
            
            if len(candidates) == 1:
                victim_name = candidates[0]
                if victim_name == "Skip":
                    self.log("\n⚖️ The town voted to Skip. No one was executed.")
                else:
                    victim = self.get_player_by_name(victim_name)
                    if victim:
                        victim.is_alive = False
                        self.log(f"\n⚖️ The town has decided. {victim.name} is executed. {victim.name}'s role was: {victim.role}")
            else:
                self.log("\n⚖️ The vote was tied. No one was executed.")
        else:
            self.log("\n⚖️ No votes were cast.")

    def check_win_condition(self) -> bool:
        mafia = [p for p in self.players if p.role == "Mafia" and p.is_alive]
        innocent = [p for p in self.players if p.role != "Mafia" and p.is_alive]

        if not mafia:
            self.log("\n🏆 GAME OVER: The Innocents have won!")
            self.is_game_over = True
            buns = input(">")
            return True
        
        if len(mafia) >= len(innocent):
            self.log("\n🏆 GAME OVER: The Mafia has taken over the town!")
            self.is_game_over = True
            buns = input(">")
            return True
        
        return False

    def start(self):
        print("Starting Mafia Simulation...\n\n")
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
