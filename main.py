# main.py
from llm import convos
from llm.ui import create_interface

def main():
    convos.load_conversations()
    demo = create_interface()
    demo.queue(max_size=20, api_open=False).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main()