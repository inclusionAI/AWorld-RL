def compute_score(reward_scores: dict[str, list[float]], ground_truth: list[list], extra_info=None, **kwargs) -> dict:

    # ---------------- 基础信息 ----------------
    user_turn_rewards = reward_scores.get("user_turn_rewards", [])
    total_interaction_rounds = len(user_turn_rewards)
    correct_tool_call = user_turn_rewards.count(-1)
    error_tool_call = user_turn_rewards.count(-2)
    error_format = user_turn_rewards.count(-3)
    is_tool_call = 1.0 if (correct_tool_call + error_tool_call) > 0 else 0.0

    format_reward =  ((total_interaction_rounds -  error_format) / total_interaction_rounds) if total_interaction_rounds > 0 else 0.0 

    tool_call_reward = correct_tool_call / (correct_tool_call + error_tool_call) if is_tool_call > 0 else 0.0

    final_reward = is_tool_call * (format_reward + tool_call_reward)

    # --------- final_reward（仍可按需改权重） ---------
    relevant = [s for s in user_turn_rewards if s == 0 or s == 1]
    progress = sum(relevant) / len(relevant) if relevant else 0.0

   
 
    return {
        "score": final_reward,
        "progress": progress,
        "format_reward":format_reward,
        "tool_call_reward":tool_call_reward,
        "is_tool_call":is_tool_call,
        "total_interaction_rounds": total_interaction_rounds
    }