from nano_graphrag import GraphRAG, QueryParam
from dotenv import load_dotenv

load_dotenv(override=True)

graph_func = GraphRAG(working_dir="./mytest")

with open("./sanguo.txt", encoding="utf-8") as f:
    graph_func.insert(f.read())

# Perform global graphrag search
print(graph_func.query("故事的主题是什么，用中文回答"))


# Perform local graphrag search (I think is better and more scalable one)
print(graph_func.query("故事的主题是什么，用中文回答?", param=QueryParam(mode="local")))

print(graph_func.query("故事的主题是什么，用中文回答?", param=QueryParam(mode="global")))