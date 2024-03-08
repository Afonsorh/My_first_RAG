from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Response
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import os
import pandas as pd
from llama_index.core.memory import ChatMemoryBuffer
from flask import Flask, request, jsonify
from llama_index.core.llms import ChatMessage

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def process_messages():
    try:
        # Get JSON data from the request
        data = request.get_json()

        #create indexes
        os.environ['OPENAI_API_KEY'] = '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'

        documents = SimpleDirectoryReader("./data/").load_data()

        # Define an LLM
        llm = OpenAI(model="gpt-3.5-turbo")

        # Build index with a chunk_size of 512
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(nodes)

        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

        chat_engine = vector_index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=(
                "You are a chatbot specialized in SAP EWM. You must only answer to questions related to SAP EWM and warehouse management." 
                "If something else than this is asked, explain that it's not your field of expertise."
            ),
        )

        # Extract messages from the JSON data
        messages = data.get("messages", [])

        # Assuming chat_history is a list
        chat_history = []
        last_message = None

        # Iterate through the messages and create ChatMessage instances
        for i, message_data in enumerate(messages):
            chat_message = ChatMessage(role=message_data.get("role", ""), content=message_data.get("content", ""))
            
            # Check if it's the last message and has a role "user"
            if i == len(messages) - 1 and chat_message.role == "user":
                last_message = chat_message.content
            else:
                chat_history.append(chat_message)

        # Prepare response

        response = chat_engine.chat(last_message, chat_history)
        return jsonify(response.response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


