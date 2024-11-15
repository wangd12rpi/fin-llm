import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def perform_batch_inference_with_metrics(context, dataset, batch_size, tokenizer, model, change_target):
    
    total_steps = len(context) // batch_size + (1 if len(context) % batch_size != 0 else 0)
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i * batch_size:(i + 1) * batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, return_token_type_ids=False)

        # ensure all tokens are moved to CUDA
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()

        # generate model predictions
        res = model.generate(**tokens, max_length=512, eos_token_id=tokenizer.eos_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]

        # extract the output text from the generated sentences
        out_text = [o.split("Answer: ")[1] if "Answer: " in o else "" for o in res_sentences]
        print(out_text)
        out_text_list += out_text

        # free up GPU memory
        torch.cuda.empty_cache()

    # update the dataset with the generated output
    dataset["out_text"] = out_text_list

    # apply the change_target function to normalize targets and predictions
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    # calculate metrics
    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average="macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average="micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average="weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted: {f1_weighted}.")

    return dataset, acc, f1_macro, f1_micro, f1_weighted
