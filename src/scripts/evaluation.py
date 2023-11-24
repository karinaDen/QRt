import os
import torch
from models.model import StableDiffusionWithControlNet
from src.scripts.qr_generator import BasicQR


# load model
model = StableDiffusionWithControlNet('cuda' if torch.cuda.is_available() else 'cpu')

# read sample prompts
prompts_path = os.path.join(os.getcwd(), 'data', 'evaluation', 'qr-prompts.txt')
with open(prompts_path, 'r', encoding='UTF-8') as file:
    prompts = list(map(str.strip, file.readlines()))

# generate images
images = []
for prompt in prompts:
    images.append(model.generate(prompt=prompt, qr_text=prompt))


# calculate number of successful encoders
correct = 0
total = 0

for img, prompt in zip(images, prompts):
    img = img.resize((128,128))
    transcribed = str(BasicQR.read(img))

    if transcribed == '-1':
        print('Failed to identify prompt:', prompt)

    total += 1
    correct += int(prompt.strip() == transcribed.strip())


print('Total number of samples:', total)
print('Correct samples:', correct)
print('Accuracy of readable results:', correct / total)
