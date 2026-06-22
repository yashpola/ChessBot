import torch
import chess
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==============================================================================
# 1. Model & Prompt Configurations
# ==============================================================================
def load_model_and_tokenizer(model_id="google/gemma-4-12b-it"):
    """Loads the Gemma 4 model and tokenizer using native system role support."""
    print(f"Loading model and tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


import re

def generate_next_move(historical_moves, white_elo, black_elo, tokenizer, model):
    """Asks Gemma 4 to generate the next best move, stopping strictly on newlines."""
    moves_history = ", ".join(historical_moves)
    
    system_message = (
        "You are an expert chess grandmaster engine. Analyze the given game state "
        "and provide the absolute best next move in Standard Algebraic Notation (SAN). "
        "Respond ONLY with the raw move token (e.g., 'e4', 'Nf3', 'O-O', 'Qxd7+'). Do not write 'thought' or give commentary."
    )
    
    user_message = (
        f"Game Context:\n- White Elo: {white_elo}\n- Black Esslo: {black_elo}\n\n"
        f"Current Move History (SAN):\n[{moves_history}]\n\n"
        "What is the next best move?"
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Force generation to halt if it attempts to write a newline or a space followed by commentary
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=12, 
            eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n")[-1]], 
            do_sample=False, 
            temperature=0.0
        )
        
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ==============================================================================
# 2. Balanced Dataset Filtering
# ==============================================================================
def get_balanced_dataset(dataset_iter, target_per_side=10):
    """Filters the streaming dataset to get exactly N white wins and N black wins."""
    white_wins = []
    black_wins = []

    print(
        f"Streaming dataset to collect {target_per_side} White wins and {target_per_side} Black wins..."
    )
    for game in dataset_iter:
        if not game.get("moves_san") or len(game["moves_san"]) < 12:
            continue

        winner = game.get("winner")
        if winner == "white" and len(white_wins) < target_per_side:
            white_wins.append(game)
        elif winner == "black" and len(black_wins) < target_per_side:
            black_wins.append(game)

        if len(white_wins) == target_per_side and len(black_wins) == target_per_side:
            break

    return white_wins + black_wins


# ==============================================================================
# 3. Interactive Engine Simulation
# ==============================================================================
def simulate_game_continuation(game, tokenizer, model, cutoff_percentage=0.6):
    """
    Simulates the game. Gemma 4 plays as the historical 'winner' starting from a
    cutoff point, while the opponent responses are pulled from the dataset.
    """

    full_history = game["moves_san"]
    cutoff_index = int(len(full_history) * cutoff_percentage)

    # Ensure it's the winner's turn at the handoff point
    target_winner = game["winner"]
    is_white_turn = cutoff_index % 2 == 0

    if (target_winner == "white" and not is_white_turn) or (
        target_winner == "black" and is_white_turn
    ):
        cutoff_index += 1

    # Reconstruct historical starting board up to the handoff cutoff
    board = chess.Board()
    current_game_history = []
    for m in full_history[:cutoff_index]:
        board.push_san(m)
        current_game_history.append(m)

    opponent_elo = game["black_elo"] if target_winner == "white" else game["white_elo"]

    records = {
        "target_winner": target_winner,
        "opponent_elo": opponent_elo,
        "historical_total_moves": len(full_history),
        "cutoff_turn": cutoff_index,
        "lm_moves_made": [],
        "dataset_moves_at_step": [],
        "outcome": "Unknown",
    }

    dataset_pointer = cutoff_index
    max_plies = 30

    for _ in range(max_plies):
        if board.is_game_over():
            break

        # --- A. GEMMA 4'S TURN ---
        raw_lm_output = generate_next_move(
            current_game_history, game['white_elo'], game['black_elo'], tokenizer, model
        )
        
        # Regular expression matching standard algebraic notation (SAN) moves,
        # filtering out alphabetical words like 'thought' completely.
        san_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]=?[QRNB]?[\+#]?|O-O-O|O-O)\b'
        match = re.search(san_pattern, raw_lm_output)
        
        if match:
            lm_move_san = match.group(1)
        else:
            # Fallback cleaning if pattern match fails entirely
            lm_move_san = raw_lm_output.split()[0].replace('thought', '').strip()
        
        expected_dataset_move = full_history[dataset_pointer] if dataset_pointer < len(full_history) else "N/A"
        records["lm_moves_made"].append(lm_move_san)
        records["dataset_moves_at_step"].append(expected_dataset_move)
        
        try:
            move_obj = board.parse_san(lm_move_san)
            board.push(move_obj)
            current_game_history.append(lm_move_san)
            dataset_pointer += 1  
        except ValueError:
            records["outcome"] = "Forfeit (Illegal Move Generated)"
            records["raw_failed_move"] = raw_lm_output 
            return records

        if board.is_game_over():
            break

        # --- B. OPPONENT'S TURN ---
        if dataset_pointer < len(full_history):
            opp_move_san = full_history[dataset_pointer]
            try:
                opp_move_obj = board.parse_san(opp_move_san)
                board.push(opp_move_obj)
                current_game_history.append(opp_move_san)
                dataset_pointer += 1
            except ValueError:
                records["outcome"] = "Diverged (Opponent move illegal on this branch)"
                return records
        else:
            records["outcome"] = "Exhausted Dataset Moves"
            return records

    # Evaluate Final Board State
    if board.is_game_over():
        result = board.result()
        if (result == "1-0" and target_winner == "white") or (
            result == "0-1" and target_winner == "black"
        ):
            records["outcome"] = "Win"
        elif result == "1/2-1/2":
            records["outcome"] = "Draw"
        else:
            records["outcome"] = "Loss"
    elif records["outcome"] == "Unknown":
        records["outcome"] = "Unfinished (Max turns reached)"

    return records


# ==============================================================================
# 4. Main Evaluation Execution
# ==============================================================================
if __name__ == "__main__":
    # Settings (10 White wins, 10 Black wins)
    GAMES_PER_SIDE = 10

    print("Connecting to Hugging Face dataset 'angeluriot/chess_games'...")
    raw_dataset = load_dataset("angeluriot/chess_games", split="train", streaming=True)

    # Collect balanced batch
    evaluation_batch = get_balanced_dataset(raw_dataset, target_per_side=GAMES_PER_SIDE)

    # Initializing with standard Gemma 4 Instruction Tuned variant
    tokenizer, model = load_model_and_tokenizer("google/gemma-4-12b-it")

    summary_stats = {"white": {"total": 0, "wins": 0}, "black": {"total": 0, "wins": 0}}

    print("\n=== Launching Interactive Evaluation (Gemma 4) ===")
    for idx, game_data in enumerate(evaluation_batch, 1):
        side = game_data['winner']
        summary_stats[side]["total"] += 1
        
        # Simulate
        log = simulate_game_continuation(game_data, tokenizer, model, cutoff_percentage=0.6)
        
        if log["outcome"] == "Win":
            summary_stats[side]["wins"] += 1
            
        # FIX 1: Safely stringify the ELO to handle NoneType values
        opp_elo_str = str(log['opponent_elo']) if log['opponent_elo'] is not None else "N/A"
            
        print(f"Game {idx:02d} | Playing As: {side.upper():<5} | Opponent ELO: {opp_elo_str:<4} | Outcome: {log['outcome']}")
        
        # FIX 2: Debugging aid to see exactly what Gemma 4 generated when it forfeited
        if "Forfeit" in log["outcome"] and log.get("raw_failed_move"):
            print(f"   ↳ [DEBUG] Model actually output: '{log['raw_failed_move']}'")

    # Final Metric Outputs
    print("\n" + "="*45)
    print("               FINAL RESULTS                  ")
    print("="*45)
    for side in ["white", "black"]:
        total = summary_stats[side]["total"]
        wins = summary_stats[side]["wins"]
        win_rate = (wins / total * 100) if total > 0 else 0.0
        print(f"As {side.upper():<5} | Games Played: {total:<2} | Wins: {wins:<2} | Win Rate: {win_rate:.1f}%")
    print("="*45)
