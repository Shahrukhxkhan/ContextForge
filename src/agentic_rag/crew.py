from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
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
	"""AgenticRag crew featuring routing, retrieval, synthesis, and verification."""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self, pdf_tool=None, web_search_tool=None, llm=None, enable_verification=True, step_callback=None, task_callback=None):
		self.pdf_tool = pdf_tool or DocumentSearchTool(file_path=default_pdf_path)
		self.web_search_tool = web_search_tool or SerperDevTool()
		self.llm = llm
		self.enable_verification = enable_verification
		self.step_callback = step_callback
		self.task_callback = task_callback

	@agent
	def router_agent(self) -> Agent:
		agent_kwargs = dict(
			config=self.agents_config['router_agent'],
			verbose=True
		)
		if self.llm:
			agent_kwargs['llm'] = self.llm
		return Agent(**agent_kwargs)

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

	@agent
	def hallucination_grader_agent(self) -> Agent:
		agent_kwargs = dict(
			config=self.agents_config['hallucination_grader_agent'],
			verbose=True
		)
		if self.llm:
			agent_kwargs['llm'] = self.llm
		return Agent(**agent_kwargs)

	@task
	def routing_task(self) -> Task:
		return Task(
			config=self.tasks_config['routing_task'],
		)

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

	@task
	def verification_task(self) -> Task:
		return Task(
			config=self.tasks_config['verification_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the full AgenticRag crew with router, retriever, synthesizer, and verifier."""
		if self.enable_verification:
			active_agents = [
				self.router_agent(),
				self.retriever_agent(),
				self.response_synthesizer_agent(),
				self.hallucination_grader_agent()
			]
			active_tasks = [
				self.routing_task(),
				self.retrieval_task(),
				self.response_task(),
				self.verification_task()
			]
		else:
			active_agents = [
				self.router_agent(),
				self.retriever_agent(),
				self.response_synthesizer_agent()
			]
			active_tasks = [
				self.routing_task(),
				self.retrieval_task(),
				self.response_task()
			]

		crew_kwargs = dict(
			agents=active_agents,
			tasks=active_tasks,
			process=Process.sequential,
			verbose=True,
		)
		if self.step_callback:
			crew_kwargs["step_callback"] = self.step_callback
		if self.task_callback:
			crew_kwargs["task_callback"] = self.task_callback

		return Crew(**crew_kwargs)
