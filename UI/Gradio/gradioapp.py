import gradio_app as gr

def read_file(file):
    with open(file.name, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

iface = gr.Interface(
    fn=read_file,
    inputs=gr.inputs.File(label="Upload a file"),
    outputs="text",
    title="File Uploader",
    description="Upload a text file to display its content."
)

iface.launch()



#%%
