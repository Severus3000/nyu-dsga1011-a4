import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (
    get_checkpoint_dir, initialize_model, initialize_optimizer_and_scheduler,
    load_model_from_checkpoint, save_model, setup_wandb,
)
from transformers import T5TokenizerFast
from load_data import get_dataloader
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--eval_only', action='store_true', help="Evaluate the saved best checkpoint on dev")
    mode.add_argument('--export_test', action='store_true', help="Export test SQL and records from the best checkpoint")

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="Stop after this many epochs without improvement (0 disables early stopping)")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help="Maximum number of tokens generated for each SQL query")
    parser.add_argument('--num_beams', type=int, default=1,
                        help="Beam search width for SQL generation")

    args = parser.parse_args()
    if not (args.eval_only or args.export_test) and args.max_n_epochs < 1:
        parser.error("--max_n_epochs must be at least 1")
    if args.patience_epochs < 0:
        parser.error("--patience_epochs must be non-negative")
    if args.num_beams < 1:
        parser.error("--num_beams must be at least 1")
    return args

def prediction_paths(args, split):
    model_type = "ft" if args.finetune else "scr"
    name = f"t5_{model_type}_{args.experiment_name}_{split}"
    return f"results/{name}.sql", f"records/{name}.pkl"

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    checkpoint_dir = get_checkpoint_dir(args)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path, model_record_path = prediction_paths(args, "dev")
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if args.patience_epochs > 0 and epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def generate_sql(model, tokenizer, encoder_input, encoder_mask, decoder_start, max_new_tokens, num_beams=1):
    generated_ids = model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        decoder_input_ids=decoder_start.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
    )
    queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [query.replace("\r", " ").replace("\n", " ").strip() for query in queries]


@torch.inference_mode()
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    queries = []

    for encoder_input, encoder_mask, decoder_input, decoder_targets, decoder_start in tqdm(dev_loader):
        encoder_input = encoder_input.to(model.device)
        encoder_mask = encoder_mask.to(model.device)
        decoder_input = decoder_input.to(model.device)
        decoder_targets = decoder_targets.to(model.device)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            use_cache=False,
        ).logits
        non_pad = decoder_targets.ne(PAD_IDX)
        total_loss += criterion(logits[non_pad], decoder_targets[non_pad]).item()
        total_tokens += non_pad.sum().item()
        del logits

        queries.extend(generate_sql(
            model, tokenizer, encoder_input, encoder_mask, decoder_start,
            args.max_new_tokens, args.num_beams,
        ))

    save_queries_and_records(queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, errors = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(bool(error) for error in errors) / len(queries)
    return total_loss / total_tokens, record_f1, record_em, sql_em, error_rate
        
@torch.inference_mode()
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    queries = []
    for encoder_input, encoder_mask, decoder_start in tqdm(test_loader):
        queries.extend(generate_sql(
            model, tokenizer, encoder_input.to(model.device), encoder_mask.to(model.device),
            decoder_start, args.max_new_tokens, args.num_beams,
        ))
    if len(queries) != len(test_loader.dataset):
        raise ValueError("Expected one generated SQL query per test question")
    save_queries_and_records(queries, model_sql_path, model_record_path)

def main():
    args = get_args()
    if args.export_test:
        model = load_model_from_checkpoint(args, best=True)
        test_loader = get_dataloader(args.test_batch_size, "test")
        sql_path, record_path = prediction_paths(args, "test")
        test_inference(args, model, test_loader, sql_path, record_path)
        print(f"Exported {len(test_loader.dataset)} test predictions to {sql_path} and {record_path}")
        return

    dev_loader = get_dataloader(args.test_batch_size, "dev")
    if not args.eval_only:
        if args.use_wandb:
            setup_wandb(args)
        train_loader = get_dataloader(args.batch_size, "train")
        model = initialize_model(args)
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        del optimizer, scheduler, model

    model = load_model_from_checkpoint(args, best=True)
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path, model_record_path = prediction_paths(args, "dev")
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

if __name__ == "__main__":
    main()
