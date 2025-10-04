input_file = "bench/questions.jsonl"
output_file = "bench/questions_fixed.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        f_out.write(line.strip() + "\n")  # strips \r and whitespace

print(f"Cleaned file written to {output_file}")
