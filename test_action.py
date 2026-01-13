from fastvideo.models.dits.matrix_game.utils import create_action_presets

actions = create_action_presets(597)

def print_summarized(actions):
    for key, tensor in actions.items():
        print(f"'{key}':")
        if tensor.ndimension() == 0:
            print(f"  {tensor}")
            continue
        
        rows = tensor.tolist()
        if not rows:
            continue
            
        def format_row(row):
            return "[" + ", ".join(f"{round(x, 4)}" for x in row) + "]"

        last_row = rows[0]
        count = 1
        for i in range(1, len(rows)):
            if rows[i] == last_row:
                count += 1
            else:
                print(f"  {format_row(last_row)} x {count}")
                last_row = rows[i]
                count = 1
        print(f"  {format_row(last_row)} x {count}")

print_summarized(actions)
