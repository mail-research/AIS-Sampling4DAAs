from vllm import LLM, SamplingParams
import jsonlines
from accelerate import Accelerator
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch
sys.path.append('/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/benchmark/RLHF-training')
from dataset.reward_dataset import *
from config import EvaluateConfig
from utils import StreamingJSONWriter
from config import Config
import os
from utils import *
from tqdm import tqdm
import tyro
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.autograd.grad_mode.set_grad_enabled(False)


GPT2_PROMPT = '''SUBREDDIT: r/relationships

TITLE: I (f/22) have to figure out if I want to still know these girls or not and would hate to sound insulting

POST: Not sure if this belongs here but it's worth a try. 

Backstory:
When I (f/22) went through my first real breakup 2 years ago because he needed space after a year of dating roand  it effected me more than I thought. It was a horrible time in my life due to living with my mother and finally having the chance to cut her out of my life. I can admit because of it was an emotional wreck and this guy was stable and didn't know how to deal with me. We ended by him avoiding for a month or so after going to a festival with my friends. When I think back I wish he just ended. So after he ended it added my depression I suffered but my friends helped me through it and I got rid of everything from him along with cutting contact. 

Now: Its been almost 3 years now and I've gotten better after counselling and mild anti depressants. My mother has been out of my life since then so there's been alot of progress. Being stronger after learning some lessons there been more insight about that time of my life but when I see him or a picture everything comes back. The emotions and memories bring me back down. 

His friends (both girls) are on my facebook because we get along well which is hard to find and I know they'll always have his back. But seeing him in a picture or talking to him at a convention having a conversation is tough. Crying confront of my current boyfriend is something I want to avoid. 

So I've been thinking that I have to cut contact with these girls because it's time to move on because it's healthier. It's best to avoid him as well. But will they be insulted? Will they accept it? Is there going to be awkwardness? I'm not sure if it's the right to do and could use some outside opinions.

TL;DR: I'm getting stronger after reading about your breakup and some friends are helping me through it but the feelings and memories bring me back down. How do I do this? My feelings have gotten worse since I've gotten into this situation but they're not stopping me from doing this. Should I cut contact with them? What are the pros and cons?<|end_of_text|>

SUBREDDIT: r/loseit

TITLE: SV & NSV! Keeping on keeping on.

POST: 30F, 5'6". SW: 236 GW: 150 CW: 219

I weigh myself weekly and measure myself monthly. I'd hit a plateau the last four weeks or so where I was stuck at 222. Felt like kind of a bummer, but knew it's because I haven't been as strict as I should with my diet, and the last week and a half have been crazy with life things, so I haven't been exercising as frequently as I've gotten used to. When I weighed myself as normal on Monday, I was kind of disappointed to see the scale not budging and figured it was time to buckle down again and really watch my diet. Today was my measure-in day, and I've felt cruddy in general since Monday because I caught some chest congestion/cold bug over the weekend. I get on the scale...it says 219. Whaaaaat? I take my measurements, which are down slightly from last month, and with an total-body loss of 8 inches from my starting point on 12/23/14! Some of my clothes have been feeling a bit looser as of late and now I know it's just not in my head. I'm now the lightest and smallest I've been since right around high school!

TL;DR: My body is so full of fat and I'm getting hit by some chest congestion/cold bug over the weekend. I'm no longer taking my measurements, which are down slightly from last month, but I still feel cruddy in general since Monday because I caught some chest congestion/cold bug over the weekend. I'm now the lightest and smallest I've been since right around high school!<|end_of_text|>

'''

GPT2_MEDIUM_PROMPT = '''SUBREDDIT: r/relationships

TITLE: Why am I so hesitant?

POST: My boyfriend (24M) and I (22F) have been together about five years.  We have lived together for about 3 months (we share a room in a house with 3 other friends).  He is a great boyfriend.  I trust him completely, he is caring and respectful, we value the same things, etc.  I know his family very well (we are from the same town) and we expect to get married eventually.  So why do I have such doubts about our relationship?

I haven't felt close to him in a very long time.  I don't feel like I can talk to him about things that are important to me, like he doesn't "get" me.  He's the happy-go-lucky type and doesn't ever see anything wrong in our relationship.  I think that if I tried to suggest ways to strengthen our relationship he would think it was ridiculous. A few months ago I tried to tell him that I feel somewhat bored and he hasn't said anything about it since.  All in all, I am just underwhelmed.

How do I deal with these feelings?  Should I try to get him to go to some sort of counseling with me?  We are very young, so sometimes I think that going to counseling or trying to "reignite the spark" is just stupid, and I should break up with him and find a new life. 

What's more important?  Security (I know he's a great guy, would be a sweet husband) or a "connection"?  I know the honeymoon phase doesn't last forever, but should I even be worrying about the future while I'm so young?  I'm a senior in college and plan on going to grad school, probably far away.  He says he'll go wherever I go, so I feel I need to put on my big girl britches and commit or drop the whole thing soon.

TL;DR: I feel like my boyfriend and I are young, close, and need to break up soon. I've been a bit hesitant to do so. What should I do to prove to him that it's OK?<|end_of_text|>

SUBREDDIT: r/AskReddit

TITLE: Let's imagine you're 45...(philosophical question)

POST: I thought of this questions a few years back and I ask it to everyone. I'm probably not the first to think of it, but oh well. 

Let's imagine you're 45, and you have all the things in life of a typical 45 year old. 

One day you are given the chance to go back in time to being 5 again, and completely re live your life. But you know everything you know now, and you can obviously use it any way you see fit. 

You can finally kiss that boy or girl you should have kissed in 9th grade. You can be a millionaire 8 year old because you invested you allowance in microsoft. You can make better use out of those few years you have with your dad before he passes away. You can finally use that perfect comeback when your boss embarrassed you. 

You can live life again exactly the same with the same family, only slightly better. Or you can set yourself up to have a completely different life. it's up to you. You know everything that's coming. 

But the catch is, on this day when you're 45 again. You'll die. No way around it. 

You still get 85 years of life, you just do them twice. 
Would you do it?

TL;DR: Imagine a 45 year old living their entire life with all the things in life of a typical 45 year old. Would you do it?<|end_of_text|>

'''


GPT2_LARGE_PROMPT = '''SUBREDDIT: r/relationships

TITLE: Me [23 M] with my girlfriend [21 F] of 1 and a half years, having trouble with giving each other space!

POST: Am I being unreasonable? We spend almost every hour together, during the day we're together mostly and in the evenings it's automatically assumed either that I am staying at hers or she is staying at mine unless otherwise previously stated. 

It's reaching finals week and things are getting stressful. I've just had one night off, I stayed at mine because I had to get up early while prior to yesterday night, we've been at eachothers everyday since Thursday. Even with last night to myself, I still didn't feel like it was truly just to myself as I wasn't feeling good. Am I a dick for asking for space or for time to myself? 

Earlier, we had an argument on the basis that as I asked if it was okay for me to stay, she says its fine, but later says that her releative has fallen very ill. It's almost as if she had mentioned it purely to manipulate me into coming to hers? I just don't know what to do. I cannot win either way.

TL;DR: I've been having trouble with giving eachother space because of argument over releative illness, we spent almost every hour together and in the evenings she says it's fine but later says its not. Am I being unreasonable?<|end_of_text|>

SUBREDDIT: r/AskReddit

TITLE: Reddit, I need help on what to do about this kid in my neighborhood.

POST: This evening I heard some disturbing news from my little brother. Apparently there is a kid down the street (11yo) who has been abusing our dogs outside when nobody is home. We keep our dogs outside because one is not properly house trained and the other tears things up if nobody is home, plus most of the time it's nice out and we leave them plenty of food and water, plus hay bedding for them to sleep in. My family has noticed that when we bring them in for the night they have been acting very skittish and will run from us if we come towards them.
Anyways, my little brother told me this evening that this boy from down the street came over to play after my brother invited him since he didn't seem to have anybody to hang out with. After a while the boy began to pick fights with my brother and his friends and repeatedly kept calling them the "N" word and insulting them. Then when my brother asked the boy to leave he told them he wasn't going to leave, he was going to beat our dogs with a bat and told him he had been doing it all week.
Now Reddit, is it wrong of me to want to kick this kids ass? Our dogs happen to be the sweetest dogs in the world, and have never harmed anything unless one of us or my siblings have been in danger. This week I am staking out my house in the daytime and watching for this kid. I know he is a minor, but I want to tan his hide red for touching my dogs. I'm wondering if I catch him should I grab him and take him to his house and tell his parents what he has been doing? Or should I just call the authorities right there?

TL;DR: Boy from down the street repeatedly hitting our dogs and threatening my family with a bat. I need help finding him and reporting him to authorities. Should I just call the police or go to his house to report?<|end_of_text|>

'''

def to_batch(x, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]

def safe_zip(*lists):
    """Zips lists together, ensuring they have the same length."""
    if not lists:
        return  # Handle empty input

    first_length = len(lists[0])
    for lst in lists[1:]:
        if len(lst) != first_length:
            raise ValueError("All lists must have the same length.")

    return zip(*lists)

if __name__=="__main__":
    config = tyro.cli(EvaluateConfig)

    world_size = torch.cuda.device_count()
    # max_model_len=4096
    policy = LLM(config.model_path, tensor_parallel_size=world_size, enable_prefix_caching=True, max_model_len=4096)
    sampling_params = SamplingParams(n=4, temperature=0.8, max_tokens=1024)
    tokenizer = policy.get_tokenizer()
    sft_dataset = globals()[f'get_{config.dataset_name}'](config.split)
    sft_dataset = sft_dataset.shuffle()
    # sft_dataset = sft_dataset.select(range(8000))
    # sft_dataset = TorchDataset(sft_dataset, tokenizer)
    path_name = config.model_path.split("/")[-1]
    print(path_name)
    
    # dataloader = DataLoader(sft_dataset, batch_size=config.batch_size, collate_fn=eval_dataset.padding_collate_fn, pin_memory=True, num_workers=8, shuffle=False)
    total_items = 0
    with jsonlines.open(f"samples/{path_name}_{config.split}_{config.dataset_name}.jsonl", "w") as writer:
        for batch in tqdm(to_batch(sft_dataset, config.batch_size), total=len(sft_dataset) // config.batch_size):
            prompts = batch['prompt']
            # conv = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
            # conv = [prompt for prompt in prompts]
            conv = prompts
            total_items += len(prompts)
            outputs = policy.generate(conv, sampling_params=sampling_params)
            outputs = [[output.outputs[idx].text.rstrip() for idx in range(len(output.outputs))] for output in outputs]
            for prompt, completion in safe_zip(conv, outputs):
                item = {
                    "prompt": prompt,
                    "completion": completion
                }
                writer.write(item)
        writer.close()
    print(f"Total items: {total_items}")