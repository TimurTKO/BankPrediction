import gradio as gr
import clickhouse_connect
from dotenv import load_dotenv
import os
load_dotenv(".env")

CLICKHOUSE_IP = os.getenv("CLICKHOUSE_IP")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER")
CLICKHOUSE_USER_PASSWORD = os.getenv("CLICKHOUSE_USER_PASSWORD")


client = clickhouse_connect.get_client(host=CLICKHOUSE_IP,
                                       username=CLICKHOUSE_USER,
                                       password=CLICKHOUSE_USER_PASSWORD)

def read_file(file):
    with open(file.name, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

iface = gr.Interface(
    fn=read_file,
    inputs=gr.File(label="Upload a file"),
    outputs="text",
    title="File Uploader",
    description="Upload a text file to display its content."
)

iface.launch()



#%%
