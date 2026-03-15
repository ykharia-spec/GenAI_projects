# GenAI_projects

Summary: This is a chat application focusing on News and general chat.
Project builds on Langgraph framework, streamlit for UX, gpt-40-mini as LLM and matplot as key components. Prompts built in to extract the best responses from LLM. There are 4 agents - router, news, general and clarify (for ambiguous query). From the user query, Router agent identifies the intent which defines the specific flow among the agents within the langgraph. As an example, if the user query intent was identified as news, the query adapted with appropriate prompt flows through the news agent that utilizes newsapi.org and the LLM to get latest news as response. Output of the response and the nodes/edges used from the graph are displayed in the browser using matplotlib. 

Need keys to access news and LLM. Create a .env with the following keys in it:

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
NEWS_API_KEY = "YOUR_NEWS_API_KEY"
