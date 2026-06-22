import torch
import chess
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# 1. Model & Prompt Configurations
# ==============================================================================
def load_model_and_tokenizer(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    print(f"Loading model and tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def generate_next_move(historical_moves, white_elo, black_elo, tokenizer, model):
    """Asks Llama-3 to generate the single next best move based on current history."""
    moves_history = ", ".join(historical_moves)
    
    system_message = (
        "You are an expert chess grandmaster. Your task is to analyze the given chess "
        "game state and recommend the absolute best next move in Standard Algebraic Notation (SAN). "
        "Respond ONLY with the next move string (e.g., 'e4', 'Nf3', 'O-O'). Do not include explanations or punctuation."
    )
    
    user_message = (
        f"Game Context:\n- White Elo: {white_elo}\n- Black Elo: {black_elo}\n\n"
        f"Current Move History (SAN):\n[{moves_history}]\n\n"
        f"What is the next best move?"
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=8, eos_token_id=terminators, do_sample=False, temperature=0.0
        )
        
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ==============================================================================
# 2. Balanced Dataset Filtering
# ==============================================================================
def get_balanced_dataset(dataset_iter, target_per_side=5):
    """Filters the streaming dataset to get exactly N white wins and N black wins."""
    white_wins = []
    black_wins = []
    
    print(f"Streaming dataset to collect {target_per_side} White wins and {target_per_side} Black wins...")
    for game in dataset_iter:
        # Filter games that are too short to evaluate meaningfully
        if not game.get('moves_san') or len(game['moves_san']) < 12:
            continue
            
        winner = game.get('winner')
        if winner == 'white' and len(white_wins) < target_per_side:
            white_wins.append(game)
        elif winner == 'black' and len(black_wins) < target_per_side:
            black_wins.append(game)
            
        if len(white_wins) == target_per_side and len(black_wins) == target_per_side:
            break
            
    return white_wins + black_wins


# ==============================================================================
# 3. Interactive Engine Simulation
# ==============================================================================
def simulate_game_continuation(game, tokenizer, model, cutoff_percentage=0.5):
    """
    Simulates the game. The LLM plays as the historical 'winner' starting from a 
    cutoff point, while the opponent responses are pulled from the dataset.
    """
    full_history = game['moves_san']
    cutoff_index = int(len(full_history) * cutoff_percentage)
    
    # Ensure it's the winner's turn at the cutoff point
    # White wins on even indices (0, 2, 4...), Black wins on odd indices (1, 3, 5...)
    target_winner = game['winner']
    is_white_turn = (cutoff_index % 2 == 0)
    
    if (target_winner == 'white' and not is_white_turn) or (target_winner == 'black' and is_white_turn):
        cutoff_index += 1
        
    # Reconstruct historical starting board up to the handoff cutoff
    board = chess.Board()
    current_game_history = []
    for m in full_history[:cutoff_index]:
        board.push_san(m)
        current_game_history.append(m)
        
    opponent_elo = game['black_elo'] if target_winner == 'white' else game['white_elo']
    
    records = {
        "target_winner": target_winner,
        "opponent_elo": opponent_elo,
        "historical_total_moves": len(full_history),
        "cutoff_turn": cutoff_index,
        "lm_moves_made": [],
        "dataset_moves_at_step": [],
        "outcome": "Unknown" 
    }
    
    # Trace dynamic indices for pulling opponent reactions
    dataset_pointer = cutoff_index
    
    # Play out up to a maximum number of bonus plies to prevent infinite loops
    max_plies = 30 
    for _ in range(max_plies):
        if board.is_game_over():
            break
            
        # --- A. MODEL'S TURN ---
        lm_move_san = generate_next_move(
            current_game_history, game['white_elo'], game['black_elo'], tokenizer, model
        )
        
        # Track dataset's alternative path at this point for similarity record-keeping
        expected_dataset_move = full_history[dataset_pointer] if dataset_pointer < len(full_history) else "N/A"
        records["lm_moves_made"].append(lm_move_san)
        records["dataset_moves_at_step"].append(expected_dataset_move)
        
        try:
            # Validate and execute the LM's move
            move_obj = board.parse_san(lm_move_san)
            board.push(move_obj)
            current_game_history.append(lm_move_san)
            dataset_pointer += 1  # Increment past our turn
        except ValueError:
            records["outcome"] = "Forfeit (Illegal Move Generated)"
            return records
            
        if board.is_game_over():
            break
            
        # --- B. OPPONENT'S TURN (Sourced dynamically from historical dataset) ---
        if dataset_pointer < len(full_history):
            opp_move_san = full_history[dataset_pointer]
            try:
                opp_move_obj = board.parse_san(opp_move_san)
                board.push(opp_move_obj)
                current_game_history.append(opp_move_san)
                dataset_pointer += 1
            except ValueError:
                # If the LM diverged significantly, historical opponent moves might become illegal
                records["outcome"] = "Diverged (Opponent move illegal on this branch)"
                return records
        else:
            # Opponent ran out of recorded historical moves
            records["outcome"] = "Exhausted Dataset Moves"
            return records

    # Evaluate Final Board State
    if board.is_game_over():
        result = board.result() # "1-0", "0-1", or "1/2-1/2"
        if (result == "1-0" and target_winner == "white") or (result == "0-1" and target_winner == "black"):
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
    # Settings
    GAMES_PER_SIDE = 10  # Evaluates 20 games total (10 White wins, 10 Black wins)
    
    print("Connecting to Hugging Face dataset 'angeluriot/chess_games'...")
    raw_dataset = load_dataset("angeluriot/chess_games", split="train", streaming=True)
    
    # Collect balanced batch
    evaluation_batch = get_balanced_dataset(raw_dataset, target_per_side=GAMES_PER_SIDE)
    tokenizer, model = load_model_and_tokenizer()
    
    summary_stats = {
        "white": {"total": 0, "wins": 0},
        "black": {"total": 0, "wins": 0}
    }
    
    game_logs = []
    
    print("\n=== Launching Interactive Evaluation ===")
    for idx, game_data in enumerate(evaluation_batch, 1):
        side = game_data['winner']
        summary_stats[side]["total"] += 1
        
        # Simulate
        log = simulate_game_continuation(game_data, tokenizer, model, cutoff_percentage=0.6)
        game_logs.append(log)
        
        if log["outcome"] == "Win":
            summary_stats[side]["wins"] += 1
            
        print(f"Game {idx:02d} | Playing As: {side.upper():<5} | Opponent ELO: {log['opponent_elo']:<4} | Outcome: {log['outcome']}")
        
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