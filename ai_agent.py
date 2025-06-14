import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    try:
        logger.info(f"Starting agent with model: {llm_id}, provider: {provider}")
        
        # Initialize LLM
        if provider == "Groq":
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            llm = ChatGroq(
                model_name=llm_id,
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            logger.info("Groq LLM initialized successfully")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Tools setup
        tools = []
        if allow_search:
            if not os.getenv("TAVILY_API_KEY"):
                logger.warning("TAVILY_API_KEY not found - search disabled")
            else:
                tools = [TavilySearchResults(
                    max_results=2,
                    api_key=os.getenv("TAVILY_API_KEY")
                )]
                logger.info("Web search tools initialized")

        # Create PROPER prompt template with required variables
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt or "You are a helpful AI assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")  # Required for agent workflow
        ])

        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            verbose=True
        )

        # Get response
        logger.info(f"Processing query: {query[:50]}...")
        response = agent_executor.invoke({"input": query})
        logger.info("Successfully generated response")
        
        return response["output"]
        
    except Exception as e:
        logger.error(f"Error in AI processing: {str(e)}", exc_info=True)
        raise Exception(f"AI processing failed: {str(e)}")