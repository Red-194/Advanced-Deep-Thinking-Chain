import os
import torch
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import concurrent.futures
from functools import partial

# LangChain imports
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    TextStreamer
)

import warnings
warnings.filterwarnings("ignore")

@dataclass
class ThoughtStage:
    """Structure for individual thought stages"""
    name: str
    content: str

@dataclass
class ThoughtProcess:
    """Complete thought process with multiple stages"""
    stages: List[ThoughtStage]
    
    def to_dict(self) -> Dict[str, str]:
        return {stage.name: stage.content for stage in self.stages}

class Config:
    """Configuration management"""
    DEFAULTS = {
        "model_name": "microsoft/phi-2",
        "temperature": 0.7,
        "max_length": 1024,
        "use_gpu": True,
        "quantize": True,
        "streaming": True,
        "max_concurrent": 2
    }
    
    def __init__(self, **kwargs):
        self.config = self.DEFAULTS.copy()
        self.config.update(kwargs)
    
    def __getattr__(self, name):
        return self.config.get(name)

class AdvancedDeepThinkingChain:
    """Enhanced deep thinking system with parallel processing"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        
        print(f"Initializing with {config.model_name} on {self.device}...")
        
        self._load_model()
        self._create_llm()
        self._create_chains()
        
        print("System initialized successfully!")
    
    def _load_model(self):
        """Load model with optimization"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None
        }
        
        if self.config.quantize and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True) if self.config.streaming else None
        )
    
    def _create_llm(self):
        """Create LLM instances"""
        llm_kwargs = {
            "pipeline": self.pipe,
            "model_kwargs": {
                "temperature": self.config.temperature,
                "max_length": self.config.max_length
            }
        }
        
        self.llm = HuggingFacePipeline(**llm_kwargs)
        
        if self.config.streaming:
            llm_kwargs["callbacks"] = [StreamingStdOutCallbackHandler()]
            self.streaming_llm = HuggingFacePipeline(**llm_kwargs)
    
    def _create_chains(self):
        """Create thinking chains with parallel capabilities"""
        # Analysis Chain
        analysis_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Analyze systematically:\n\nQuestion: {question}\n\nProvide:\n1. Key concepts\n2. Question type\n3. Assumptions\n4. Sub-questions\n\nAnalysis:"""
        )
        self.analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt, output_key="analysis")
        
        # Research Chain
        research_prompt = PromptTemplate(
            input_variables=["question", "analysis"],
            template="""Research perspectives:\n\nQuestion: {question}\nAnalysis: {analysis}\n\nExplore:\n1. Viewpoints\n2. History\n3. Debates\n4. Applications\n5. Implications\n\nResearch:"""
        )
        self.research_chain = LLMChain(llm=self.llm, prompt=research_prompt, output_key="research")
        
        # Critique Chain
        critique_prompt = PromptTemplate(
            input_variables=["question", "analysis", "research"],
            template="""Critically evaluate:\n\nQuestion: {question}\nAnalysis: {analysis}\nResearch: {research}\n\nExamine:\n1. Assumptions\n2. Fallacies\n3. Evidence\n4. Limitations\n5. Alternatives\n\nCritique:"""
        )
        self.critique_chain = LLMChain(llm=self.llm, prompt=critique_prompt, output_key="critique")
        
        # Creative Chain
        creative_prompt = PromptTemplate(
            input_variables=["question", "analysis", "research"],
            template="""Generate insights:\n\nQuestion: {question}\nAnalysis: {analysis}\nResearch: {research}\n\nProvide:\n1. Connections\n2. Metaphors\n3. Solutions\n4. Experiments\n5. Perspectives\n\nCreative:"""
        )
        self.creative_chain = LLMChain(llm=self.llm, prompt=creative_prompt, output_key="creative")
        
        # Synthesis Chain
        synthesis_prompt = PromptTemplate(
            input_variables=["question", "analysis", "research", "critique", "creative"],
            template="""Synthesize:\n\nQuestion: {question}\nAnalysis: {analysis}\nResearch: {research}\nCritique: {critique}\nCreative: {creative}\n\nAnswer:\n1. Addresses question\n2. Integrates insights\n3. Acknowledges nuance\n4. Gives examples\n5. Offers implications\n\nSynthesis:"""
        )
        self.synthesis_chain = LLMChain(
            llm=self.streaming_llm if self.config.streaming else self.llm,
            prompt=synthesis_prompt,
            output_key="synthesis"
        )
        
        # Parallel execution for independent chains
        self.parallel_chains = RunnableParallel(
            critique=self.critique_chain,
            creative=self.creative_chain
        )
        
        # Conversation chain
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=ConversationBufferMemory(),
            verbose=False
        )
    
    async def _run_chain_async(self, chain, inputs):
        """Run a chain asynchronously"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, lambda: chain.run(inputs))
    
    async def think_deeply(self, question: str) -> ThoughtProcess:
        """Process question with parallel execution where possible"""
        print("\n🧠 Starting Deep Thinking...\n")
        
        stages = []
        
        # Sequential steps
        analysis = await self._run_chain_async(self.analysis_chain, {"question": question})
        stages.append(ThoughtStage("analysis", analysis))
        
        research = await self._run_chain_async(self.research_chain, {
            "question": question,
            "analysis": analysis
        })
        stages.append(ThoughtStage("research", research))
        
        # Parallel steps for critique and creative
        critique_task = self._run_chain_async(self.critique_chain, {
            "question": question,
            "analysis": analysis,
            "research": research
        })
        creative_task = self._run_chain_async(self.creative_chain, {
            "question": question,
            "analysis": analysis,
            "research": research
        })
        
        critique, creative = await asyncio.gather(critique_task, creative_task)
        stages.extend([ThoughtStage("critique", critique), ThoughtStage("creative", creative)])
        
        # Final synthesis
        synthesis = await self._run_chain_async(self.synthesis_chain, {
            "question": question,
            "analysis": analysis,
            "research": research,
            "critique": critique,
            "creative": creative
        })
        stages.append(ThoughtStage("synthesis", synthesis))
        
        return ThoughtProcess(stages)
    
    async def quick_think(self, question: str) -> str:
        """Quick response"""
        prompt = f"Answer thoughtfully:\n\nQuestion: {question}\n\nAnswer:"
        return await self._run_chain_async(self.llm, prompt)
    
    async def follow_up(self, question: str) -> str:
        """Follow-up with conversation context"""
        return await self._run_chain_async(self.conversation_chain, {"input": question})

class InteractiveAdvancedThinker:
    """Enhanced interactive interface"""
    
    def __init__(self):
        self.system = None
        self.history = []
    
    def setup(self):
        """Interactive setup with proper model selection"""
        print("🤖 Advanced Deep Thinking System")
        print("=" * 50)
        
        config = Config()  # Assuming Config is a class defined elsewhere
        
        # Define available models with descriptions for clarity
        models = {
            "1": ("microsoft/phi-2", "2.7B parameters - Balanced"),
            "2": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B parameters - Fast"),
            "3": ("google/gemma-2b-it", "2B parameters - Good quality"),
            "4": ("Qwen/Qwen1.5-0.5B-Chat", "0.5B parameters - Very fast"),
            "5": ("custom", "Enter your own model")
        }
        
        # Display model options
        print("\nAvailable models:")
        for key, (name, desc) in models.items():
            if key != "5":
                print(f"{key}. {name} ({desc})")
            else:
                print(f"{key}. Custom model")
        
        # Get user choice with default to "1"
        choice = input("\nSelect model (1-5, default 1): ").strip() or "1"
        
        # Map the choice to the model name
        if choice == "5":
            model_name = input("Enter custom HuggingFace model name: ").strip()
            while not model_name:  # Ensure a non-empty custom name
                print("Model name cannot be empty.")
                model_name = input("Enter custom HuggingFace model name: ").strip()
        elif choice in models:
            model_name = models[choice][0]  # Extract the model name
        else:
            model_name = models["1"][0]
            print(f"Invalid choice. Using default: {model_name}")
        
        config.config["model_name"] = model_name
        
        # Streaming option
        streaming = input("Enable streaming? (y/n, default n): ").strip().lower() == 'y'
        config.config["streaming"] = streaming
        
        # Temperature option
        try:
            temp = float(input("Temperature (0.1-1.0, default 0.7): ").strip() or "0.7")
            config.config["temperature"] = max(0.1, min(1.0, temp))
        except ValueError:
            print("Invalid temperature. Using default 0.7")
            config.config["temperature"] = 0.7
        
        # Initialize the system
        print(f"\nInitializing with {model_name} on {config.device}...")
        self.system = AdvancedDeepThinkingChain(config)  # Assuming this class exists
    
    async def run(self):
        """Run interactive loop"""
        if not self.system:
            self.setup()
        
        print("\n✨ Commands: 'deep', 'quick', 'follow', 'history', 'exit'")
        
        mode = "deep"
        while True:
            try:
                user_input = input(f"\n[{mode}] Question: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() in ['deep', 'quick', 'follow']:
                    mode = user_input.lower()
                    continue
                
                if user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                if not user_input:
                    continue
                
                if mode == "deep":
                    thought = await self.system.think_deeply(user_input)
                    self.history.append({"question": user_input, "thought": thought, "mode": "deep"})
                    print("\n=== SYNTHESIS ===")
                    print(thought.stages[-1].content)
                
                elif mode == "quick":
                    response = await self.system.quick_think(user_input)
                    self.history.append({"question": user_input, "response": response, "mode": "quick"})
                    print("\n=== RESPONSE ===")
                    print(response)
                
                elif mode == "follow":
                    response = await self.system.follow_up(user_input)
                    self.history.append({"question": user_input, "response": response, "mode": "follow"})
                    print("\n=== FOLLOW-UP ===")
                    print(response)
                    
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def _show_history(self):
        """Show thinking history"""
        if not self.history:
            print("\n📚 No history yet.")
            return
        
        print("\n📚 HISTORY:")
        for i, entry in enumerate(self.history, 1):
            print(f"\n{i}. Question: {entry['question']}")
            print(f"   Mode: {entry['mode']}")
            if entry['mode'] == 'deep':
                print(f"   Synthesis: {entry['thought'].stages[-1].content[:100]}...")
            else:
                print(f"   Response: {entry['response'][:100]}...")

async def main():
    """Async main entry"""
    thinker = InteractiveAdvancedThinker()
    await thinker.run()

if __name__ == "__main__":
    asyncio.run(main())
