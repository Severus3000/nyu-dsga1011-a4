# DS-GA 1011: Homework 4 Part 2

> ### Please start early.

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment. Note that Part 2 pins different package versions than Part 1, so use a separate environment from Part 1.

Example of virtual environment creation using conda:
```bash
conda create -n dsga1011-hw4-part2 python=3.11
conda activate dsga1011-hw4-part2
python -m pip install -r requirements.txt
```

You can refer to the [HPC Tutorials](https://github.com/Athul-R/NYU-HPC-Tutorials) for more information on how to use the NYU HPC.

## Evaluation commands

If you have saved predicted SQL queries and associated database records, you can compute F1 scores using:
```bash
python evaluate.py
  --predicted_sql results/t5_ft_experiment_dev.sql
  --predicted_records records/t5_ft_experiment_dev.pkl
  --development_sql data/dev.sql
  --development_records records/ground_truth_dev.pkl
```

## Report and export the selected checkpoint

For the final configuration, place the epoch 6 checkpoint files in
`checkpoints/ft_experiments/experiment/best/`. The selected decoder uses beam
width 4 and a 512-token generation limit. To regenerate the development
predictions and save the Q6 metrics and error examples:

```bash
python train_t5.py --finetune --experiment_name experiment --eval_only --num_beams 4
python evaluate.py \
  --predicted_sql results/t5_ft_experiment_dev.sql \
  --predicted_records records/t5_ft_experiment_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl \
  --development_questions data/dev.nl > results/t5_ft_experiment_dev_analysis.json
```

The error categories describe execution outcomes. Compare the example queries
with their questions and reference SQL to explain the causes in the report.
Once you have selected the final model on the development set, export the Q7
files in test-set order:

```bash
python train_t5.py --finetune --experiment_name experiment --export_test --num_beams 4
```

## Submission

You need to submit your test SQL queries and their associated SQL records. Please only submit your final files corresponding to the test set.

For SQL queries, ensure that the name of the submission files (in the `results/` subfolder) are:
- `t5_ft_experiment_test.sql` (for extra credit `t5_ft_experiment_ec_test.sql`)

For database records, ensure that the name of the submission files (in the `records/` subfolder) are:
- `t5_ft_experiment_test.pkl` (for extra credit `t5_ft_experiment_ec_test.pkl`)

⚠️ Note that the predictions in each line of the .sql file or in each index of the list within the .pkl file must match each natural language query in `data/test.nl` in the order they appear.
