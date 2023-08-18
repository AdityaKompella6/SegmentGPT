## seggpt
Training and Inference Code for SegGPT

## Installation
Run wget https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth to get model

pip install -r requirements.txt for libraries

## Usage

# Prediction:

Single Image inference:
```
import seggpt_inference
##Prompt image: PIL image
##Prompt Mask: Np array binary mask
##test image: PIL image
out_mask,out_image = seggpt_inference.predict(prompt_img,prompt_mask,test_image,threshold = 120)
##out mask is binary mask
##out image is the rgb mask before pre-processing
```
Batched Image inference:
```
import seggpt_inference
##Prompt images: list of PIL image
##Prompt Masks: list of Np array binary mask
##test images: list of PIL image
out_masks,out_images = seggpt_inference.predict_batch(prompt_imgs,prompt_masks,test_images,threshold = 120)
##out mask is binary mask
##out image is the rgb mask before pre-processing
```
Tiled Image inference:
```
import seggpt_inference
##Prompt image: PIL image
##Prompt Mask: Np array binary mask
##test image: PIL image
out_mask = seggpt_inference.predict_tiled(prompt_img,prompt_mask,test_image)
##out mask is binary mask
```

Fine-tuned Inference Tiled:
```
import seggpt_inference
##Prompt image: torch tensor
##Prompt Mask: torch tensor
##test image: PIL image
out_mask = seggpt_inference.predict_tiled_finetuned(prompt_img,prompt_mask,test_image)  
##out mask is binary mask
```


See all .ipynb files to see prediction examples


# Finetune:

```
from seggptdataset import SegTensorDataset

ds = SegTensorDataset(prompt_imgs, prompt_masks, 0.2)

train_dl = DataLoader(ds, batch_size=2, pin_memory=True)

val_dl = DataLoader(ds, batch_size=1, pin_memory=True)

cat_to_class = {0: "Background", 1: "Wires"}

from seggptICLfinetune import fine_tune

from models_seggpt import LearnablePrompt

prompt = LearnablePrompt()
prompt.load_state_dict(torch.load("bestwires_learned_promptv3.pt"))
torch.cuda.empty_cache()

final_prompt, losses = fine_tune(
    train_dl,
    val_dl,
    color_to_cat,
    cat_to_class,
    prompt,
    "bestwires_learned_promptv4.pt",##Name to save prompt
    "seggpt-wires",##wandb_project_name
    1000,##Num_epochs
    50,##Warmup_epochs
    1e-4,##Start_lr
    1e-6,##End_lr
    accum_iters=4,
)

torch.save(final_prompt.state_dict(), "wires_learned_promptv4.pt")```
```
See segmentationfinetunemainwires.py for dataset creation/training example
