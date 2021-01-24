import numpy as np
import os
import torch
import random
from pathlib import Path
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from utils import get_best_indexes
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, squad_evaluate
import timeit
import json


class Trainer():
    def __init__(self, args, model, logger, device, device_ids, words_dict=None):
        self.model = model
        self.args = args
        self.logger = logger
        self.device = device # this class is torch.device 
        self.device_ids = device_ids
        self.words_dict = words_dict
        self.labels = ["start", "end"]

        # self.writer = TensorboardWriter(config.log_dir, self.logger, True)



    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if len(self.device_ids) > 0:
            torch.cuda.manual_seed_all(seed)



    def train(self, args, train_dataset, eval_dataset, tokenizer):
        if args["local_rank"] in [-1, 0]:
            tb_writer = SummaryWriter(logdir="test_runs/" + os.path.basename(args["save_dir"]))


        train_batch_size = args["per_gpu_train_batch_size"] * max(1, len(self.device_ids))
        train_sampler = (
            RandomSampler(train_dataset)
            if args["local_rank"] == -1
            else DistributedSampler(train_dataset)
        )
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        if args["max_steps"] > 0:
            t_total = args["max_steps"]
            args["num_train_epochs"] = (
                args["max_steps"]
                // (len(train_dataloader) // args["gradient_accumulation_steps"]) + 1)
        else:
            t_total = (
                len(train_dataloader)
                // args["gradient_accumulation_steps"] * args["num_train_epochs"])

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in self.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total)


        # multi-gpu training (should be after apex fp16 initialization)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if args["local_rank"] != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args["local_rank"]],
                output_device=args["local_rank"],
                find_unused_parameters=True,)

        if args["is_testing"]:
            self.logger.info("***** Running testing *****")
            self.logger.info("  Num examples = %d", len(eval_dataset))
            
            eval_result, start_result, end_result = self.evaluate(args, tokenizer, eval_dataset, self.words_dict)
            self.logger.info("***** Finish testing *****")
            exit()




        # Start training now!!!!!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataset))
        self.logger.info("  Num Epochs = %d", args["num_train_epochs"])
        self.logger.info("  Instantaneous batch size per GPU = %d", args["per_gpu_train_batch_size"])
        self.logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size * args["gradient_accumulation_steps"] * (torch.distributed.get_world_size() if args["local_rank"] != -1 else 1),)
        self.logger.info("  Gradient Accumulation steps = %d", args["gradient_accumulation_steps"])
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["local_rank"] not in [-1, 0])
        self.set_seed(args["seed"])  # Added here for reproductibility (even between python 2 and 3)

        
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args["local_rank"] not in [-1, 0])
        
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                

                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "start_positions": batch[5],
                    "end_positions":batch[6],
                    "bbox": batch[4],
                    "token_type_ids": batch[2],
                }

                outputs = self.model(**inputs)
                loss = outputs[0]

                if len(self.device_ids) > 1:
                    loss = loss.mean()
                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args["max_grad_norm"])
                    scheduler.step()
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

                    if(args["local_rank"] in [-1, 0] and args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0):
                        
                        if(args["local_rank"] == -1 and args["evaluate_during_training"] and global_step % args["eval_steps"] == 0):
                            eval_result, start_result, end_result = self.evaluate(args, tokenizer, eval_dataset)
                            tb_writer.add_scalar("eval_sentence", eval_result, global_step)
                            tb_writer.add_scalar("eval_start", start_result, global_step)
                            tb_writer.add_scalar("eval_end", end_result, global_step)

                            '''
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            '''
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss",(tr_loss - logging_loss) / args["logging_steps"], global_step,)
                        logging_loss = tr_loss

                    # save model checkpoint
                    if(args["local_rank"] in [-1, 0] and args["save_steps"] > 0 and global_step % args["save_steps"] == 0):
                        output_dir = os.path.join(args["save_dir"], "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        self.logger.info("Saving model checkpoint to %s", output_dir)

            if args["max_steps"] > 0 and global_step > args["max_steps"]:
                    epoch_iterator.close()
                    break

            if args["max_steps"] > 0 and global_step > args["max_steps"]:
                epoch_iterator.close()
                break

        if args["local_rank"] in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss/global_step


    def to_list(self, tensor):
        return tensor.detach().cpu().tolist()


    def evaluate(self, args, tokenizer, eval_dataset, words_dict=None, prefix=""):
        # words_dict is only for testing purpose
        eval_batch_size = args["per_gpu_eval_batch_size"] * max(1, len(self.device_ids))
        
        # set to distributed gpus or not
        eval_sampler = (SequentialSampler(eval_dataset) if args["local_rank"] == -1 else DistributedSampler(eval_dataset))

        eval_data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        # do evaluation
        self.logger.info("***** Running evaluation %s *****", prefix)
        self.logger.info("  Num examples = %d", len(eval_dataset))
        self.logger.info("  Batch size = %d", args["eval_batch_size"])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        all_results = []
        start_time = timeit.default_timer()
        self.model.eval()
        correct = 0
        wrong = 0
        start_correct = 0
        end_correct = 0
        start_wrong = 0
        end_wrong = 0

        test_result = {}
        final_output = []


        for batch in tqdm(eval_data_loader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "bbox": batch[4],
                    "token_type_ids": batch[3],
                }
                start_positions = self.to_list(batch[5])
                end_positions = self.to_list(batch[6])
                outputs = self.model(**inputs)
                quesiton_ids = batch[7]
                unique_ids = batch[8]
                # words_list = batch[9]
                location_list = batch[9]



            for i, ids in enumerate(unique_ids):

                output = [self.to_list(output[i]) for output in outputs]
                start_logits, end_logits = output

                start = get_best_indexes(start_logits)[0]
                end = get_best_indexes(end_logits)[0]

                if args["is_testing"]:
                    start_ground = location_list[i][start]
                    end_ground = location_list[i][end]

                    q_id = quesiton_ids[i].item()
                    words_list = words_dict[q_id]
                    # print('###########')
                    # print(start_ground)
                    # print(end_ground)
                    
                    if (start_ground > end_ground) or (start_ground < 0 or end_ground < 0):
                        if test_result.__contains__(q_id):  # already exist
                            continue                     
                        else:                               # add new
                            test_result[q_id] = ""                     
                    else:
                        ground_words = words_list[start_ground : end_ground + 1]
                        text = ground_words[0]
                        for i in range(len(ground_words) - 1):
                            text = text + " " + ground_words[i+1]
                        
                        test_result[q_id] = text

                else:
                    if start_positions[i] != 0 and end_positions[i] != 0:
            
                        if start_positions[i] == start:
                            start_correct += 1
                        else: 
                            start_wrong += 1

                        if end_positions[i] == end:
                            end_correct += 1
                        else:
                            end_wrong += 1

                        if (start_positions[i] == start and end_positions[i] == end):
                            correct += 1
                        else:
                            wrong += 1



        if args["is_testing"]:
            # process final result
            for key in test_result.keys():
                final_output.append({"questionId": key, "answer": test_result[key]})

            output_path = Path(args["eval_output_dir"])
            output_path.mkdir(parents=True, exist_ok=True)
            result_file = output_path.joinpath(Path('test_result.json'))
            
            with result_file.open(mode='w') as f:
                json.dump(final_output, f)

            self.logger.info("Test output saved")
            self.model.train()
            return -1, -1, -1

        else:
            result = correct / (correct + wrong)
            start_result = start_correct / (start_correct + start_wrong)
            end_result = end_correct / (end_correct + end_wrong)

            self.logger.info("Accuracy for each sentence is: {}".format(result))
            self.logger.info("Accuracy for each start is: {}".format(start_result))
            self.logger.info("Accuracy for each end is: {}".format(end_result))

            self.model.train()
            return result, start_result, end_result




