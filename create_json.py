import json
import argparse
import random
import numpy as np

from generate import generate

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--num", type=int, default=1000, help="Task examples to generate.")
    parser.add_argument("--cont", type=float, default=0.5, help="Percentage of how many examples are allowed to contain the label.")
    parser.add_argument("--frac_invalid", type=float, default=0.)
    parser.add_argument("--num_ops", type=int, default=5)
    args = parser.parse_args()
    return args

def generate_json(num, cont, num_ops, frac_invalid):
    """
    Generates json files with examples
    args:
      num: number of examples
      cont: how many of the examples are allowed to contain the label (i.e. the output matrix shape) in their string
    """
    max_cont = int(num * cont)
    num_invalid = int(num * frac_invalid)
    invalid_choices = np.random.choice(num, num_invalid, replace=False)

    data = {}
    data["examples"] = []

    cur_cont = 0
    while len(data["examples"]) < num:
      invalid = cur_cont in invalid_choices
      input, target, contained = generate(num_ops=num_ops, invalid=invalid)

      # Handle contained examples / confounders
      if (contained) and (cur_cont >= max_cont):
        continue
      else:
        cur_cont += 1

      data["examples"].append({"input": input, "target": target})
    
    # Shuffle to avoid having harder examples at the end due to skipping
    random.shuffle(data["examples"])

    with open('task.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)
 

if __name__ == "__main__":
    args = parse_args()
    generate_json(args.num, args.cont, args.num_ops, args.frac_invalid)
