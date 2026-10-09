from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai_tools import PDFSearchTool
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure package or direct import resolves correctly
try:
    from .tools.custom_tool import DocumentSearchTool
except (ImportError, ValueError):
    try:
        from agentic_rag.tools.custom_tool import DocumentSearchTool
    except ImportError:
        from src.agentic_rag.tools.custom_tool import DocumentSearchTool

# Resolve knowledge/dspy.pdf dynamically relative to ContextForge root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
default_pdf_path = str(ROOT_DIR / "knowledge" / "dspy.pdf")

@CrewBase
class AgenticRag():
	"""AgenticRag crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self, pdf_tool=None, web_search_tool=None, llm=None):
		super().__init__()
		self.pdf_tool = pdf_tool or DocumentSearchTool(file_path=default_pdf_path)
		self.web_search_tool = web_search_tool or SerperDevTool()
		self.llm = llm

	@agent
	def retriever_agent(self) -> Agent:
		agent_kwargs = dict(
			config=self.agents_config['retriever_agent'],
			verbose=True,
			tools=[t for t in [self.pdf_tool, self.web_search_tool] if t]
		)
		if self.llm:
			agent_kwargs['llm'] = self.llm
		return Agent(**agent_kwargs)

	@agent
	def response_synthesizer_agent(self) -> Agent:
		agent_kwargs = dict(
			config=self.agents_config['response_synthesizer_agent'],
			verbose=True
		)
		if self.llm:
			agent_kwargs['llm'] = self.llm
		return Agent(**agent_kwargs)

	@task
	def retrieval_task(self) -> Task:
		return Task(
			config=self.tasks_config['retrieval_task'],
		)

	@task
	def response_task(self) -> Task:
		return Task(
			config=self.tasks_config['response_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AgenticRag crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks,   # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
