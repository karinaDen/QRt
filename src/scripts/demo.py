import gradio
from ...models.model import StableDiffusionWithControlNet


model = StableDiffusionWithControlNet('cuda')

demo = gradio.Interface(
    fn=model.generate,
    inputs=["text","text"],
    outputs="image",
    title="QR Code Generator",
    description="Enter the text for which you want to generate a QR code.",
    allow_flagging="never",

)

demo.launch()
